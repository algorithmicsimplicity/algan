# Do the rank-3/rank-4 proofs certify anything on real scenes?

**Rank 3: no. Zero occlusion rejections on seven of seven scenes.**
**Rank 4: barely. It fires on two of seven, at 2.0% and 0.75% of covered pixels.**

The counters are scene-deterministic and device-independent, so this was taken
on CPU. Every scene in `tests/full_renders/scenes/` rendered at `PREVIEW`
under that suite's own settings contract (pinned 1.5 GiB frame-window split,
eager torch), with both switches on, plus the `nn` anchor from the A/B script.
Script: `benchmarks/performance/tile_proof_sweep.py`.

## The answer

| scene | triangles | circuits | certified occluders | **occlusion rejections** | **simple pixels** | covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shapes_and_timeline` | 131,219 | 4,348 | 86 | **0** | **185,393** | 9,075,458 (2.0%) |
| `complex_hierarchy_become` | 17,632 | 440 | 32 | **0** | **12,938** | 1,724,004 (0.75%) |
| `materials_and_lighting` | 387,408 | 2,640 | 39,130 | **0** | 0 | 9,727,932 |
| `solids_and_camera` | 229,968 | 2,560 | 46 | **0** | 0 | 5,604,431 |
| `text_and_media` | 909,380 | 1,995 | 0 | **0** | 0 | 4,309,328 |
| `manim_compat_and_plots` | 0 | 872 | 0 | **0** | 0 | 2,593,453 |
| `nn` (the A/B anchor) | 25,082 | 41 | 0 | **0** | 0 | 34,878,832 |

## Rank 3 fails twice over, and the second failure is the interesting one

**Most candidates cannot even be occluders.** A candidate must fill its whole
16x16 tile before the containment proof is worth attempting, and almost none
do: `fills_tile` is 1,016 of 549,722 triangle candidates on
`shapes_and_timeline` (0.18%), 6,052 of 8,315,603 on `text_and_media` (0.07%),
26,887 of 2,691,921 on `solids_and_camera` (1.0%). Algan's triangles are
smaller than a tile, because they come from tessellated surfaces and glyph
outlines rather than from hand-authored quads.

**But where it does certify, the rejection still never fires.**
`materials_and_lighting` certified **39,130 occluders** — 1.5% of its
candidates, a real population — and rejected **nothing**. That is not the
`fills_tile` gate; the occluders are there. It is the second half of the
proof: rejecting a candidate needs its own distance interval to close over the
tile and land strictly behind the occluder's bound in a *different* depth bin,
and `_rect_proof`'s interval arithmetic is too wide to produce a usable `near`
over a 16x16 rectangle for geometry at this scale. The same measurement on the
anchor put it at 58 finite far-bounds out of 93,070 candidates.

So Rank 3's occlusion mechanism has now been observed to fire on exactly one
scene: the `overdraw` fixture, built backwards from the gates out of flat open
frame-filling quads. Nothing in the corpus resembles it.

## Rank 4 fires, and it does not matter

Two scenes certify. The rate is the problem: 2.0% and 0.75% of covered pixels.
On the `overdraw` fixture, certifying **99.9%** of pixels cut `compact_sheets`
by two thirds and bought ~2% end-to-end, because compaction is ~8% of that
render. Scaling that to 2% of pixels leaves about 0.04% — three orders of
magnitude below the +0.4% the switch costs on the anchor.

The per-pixel conjunction is what flattens it, and `materials_and_lighting`
shows it cleanly: 8,867,643 fragments pass the per-fragment pre-gates, 71,166
survive the geometry and closed-shell proof, and **0 pixels** qualify, because
a pixel is simple only if *every* one of its layers is. `solids_and_camera`
gets 2,214,033 fragments through the pre-gates and 0 through the shell gate —
96.7% of its triangles are closed shells, which `interior_fragment_proofs`
rejects outright.

## What this settles

Both mechanisms are structurally incompatible with how Algan builds geometry,
not merely untuned:

* tessellated surfaces and glyph outlines produce triangles far smaller than a
  screen tile, so tile-filling occluders essentially do not occur;
* closed shells are the default for Algan's 3-D primitives, and Rank 4 rejects
  them outright;
* the interval arithmetic that makes the proofs sound is too conservative over
  a tile-sized rectangle to yield a usable distance bound at these scales.

None of those is a parameter. Widening the tile makes the intervals worse
(`tiled_primary_ab_3.md`); narrowing it makes tile-filling rarer still.

Measured cost on the anchor, after four rounds of optimisation:
`raster_tile_binning` +1.03%, `raster_simple_interiors` +0.4%.
Measured benefit on the corpus: none.
