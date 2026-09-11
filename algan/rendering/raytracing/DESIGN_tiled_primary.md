# Opt-in tiled primary discovery and simple interiors

This implements the migration stage of the speed audit's **Ranks 3 and 4**:
change candidate generation and simple-pixel metadata while retaining the
existing analytic acceptance kernels and sheet/material/shadow/continuation
consumer. The corrected audit's observation matters here: sheets are often
nearly as numerous as fragments. No reduction in that ratio is assumed.

## Settings and compatibility

```python
from algan import SETTINGS

SETTINGS.raytracing.experimental.set(
    raster_tile_binning=True,
    raster_simple_interiors=True,
)
```

Both default to `False`. They are independent runtime settings, captured once
per coverage window: either can be enabled without the other. No environment
variables, compiler patch, Q1/Q2 dispatch implementation, or runtime reset is
required. Set both to `False` to restore the original frontend/compactor.

The implementation uses existing portable Quadrants ndarray operations and
exact integer gather wrappers, without subgroup-width assumptions, hardware
rasterization, or native ray-query requirements. This does **not** add AMD
backend selection/adoption to an Algan runtime that does not already support
it. CPU validation does not establish CUDA, Metal, or AMD hardware parity.

## Rank 3: two-level bins and pre-emission rejection

`tile_raster.tiled_specs` constructs conservative per-frame primitive records
from the same precomputed triangle/circuit bounds as the reference frontend.
Each record is inserted into intersected **64x64 coarse bins**. A coarse CSR
list feeds its sixteen **16x16 fine tiles**, including partial tiles at frame
edges. These sizes are implementation choices, not measured optimal values.

Fine candidates own disjoint intersections of a tile with a primitive bbox.
The count/write kernels therefore emit a primitive at a pixel at most once,
even across coarse boundaries or a nonzero frame-window start. Circuits,
near-plane straddlers, and uncertain geometry remain ordinary candidates.
The existing triangle/circuit coverage kernels still decide acceptance and
write the same barycentrics, analytic areas, masks, and references.

Before those kernels run, a triangle candidate can be rejected in two ways:

* **Geometric separation:** the triangle is outside an expanded tile in both
  floating-point and snapped fixed-point edge representations. Expansion by
  one pixel conservatively includes the current coverage filter's reach.
* **Opaque occlusion:** another triangle strictly contains the whole tile,
  is definitely materially opaque with zero transmission, and has a certified
  maximum hit distance strictly before the candidate's minimum distance.

A sample mask, near-one area, center depth, or vertex minimum alone is never an
occlusion certificate. Custom fragment pipelines disable early material
occlusion. Nonconstant opacity/transmission maps are unproven. Constant 1x1
promoted materials use the same material readers as rendering; alpha must be
**exactly 1** and transmission **exactly 0**, not the existing approximate
opaque-class thresholds. Frame-animated material data is checked per frame.

`_rect_proof` separately checks strict half-plane containment for the unsnapped
area triangle and the snapped ownership triangle. Float interval operations
round endpoints outward, including a normal-sized bound around values that
may flush to zero. Reciprocal and square-root estimates are widened and then
verified with outward product inequalities, rather than assuming a backend's
fast-math approximation is correctly rounded. A failed check is unproven.
Rational perspective interpolation and distance are bounded
over the **entire rectangle**. Denominators spanning zero, nonfinite results,
near-camera ambiguity and unsafe fixed-point coordinate ranges fail proof.
Loose intervals reduce eligibility rather than authorize a rejection.

Distance separation also excludes depth-bin ties and saturated bins. The
reference's late opaque-prefix rule is retained unchanged for other cases.
Zero-sample area donors and crossing candidates are not discarded merely for
having no ownership samples or an apparently farther center.

### Chunk packing: box or rows

A surviving candidate's box is its bbox clipped to the tile, and rejecting a
tile says nothing about how full the tiles that survive are. A thin diagonal
crosses a tile while owning two or three of its pixels per row, so emitting
the box hands the COUNT pass the whole tile. Measured on the `nn` scene at HD:
11.4M candidate pixels against the reference frontend's 2.9M for the same
frame, which is what `raster_span_candidates` is for there.

So a candidate may instead emit one `bh == 1` box per pixel row, clipped to
the triangle's own conservative x-extent over that row *and* to the tile — the
reference's `_span_row_extent` / `_span_mode` helpers, behind the same
`raster_span_candidates` kill switch and the same minimum box area. The tile
bounds the span, so two tiles still never claim the same pixel, and the row
extent carries the reference's one-pixel margin, so the pixel set the geometry
kernels see is unchanged.

Rows are not always better. A tile a large triangle covers outright is already
packed at `raster_chunk` pixels a chunk, and splitting it by row only
multiplies the chunk count (measured: the overdraw scene went 3.08s -> 3.99s
warm on CPU when every eligible candidate took rows). The COUNT pass therefore
walks the rows, compares their pixels against the box's, and takes rows only
where they at least **halve** the candidate pixels. That decision rides in
candidate `flags` bit 2 (`_SPAN_FORM_BIT`) so the WRITE pass reads it back
rather than re-deriving it and disagreeing with the slice it must fill.

WRITE is driven by the COUNT pass's prefix slice in both forms. A rejected
candidate counted zero chunks and must emit nothing: walking its geometry
anyway overwrites the next candidate's slice and runs off the buffer.

### Ordering and capacity

After emission, sparse tile-owned pixel buckets form a pixel CSR by histogram,
scan and scatter. Only covered **pixel IDs**, not the complete fragment stream,
are globally ordered. Each pixel's fragments use the existing insertion/
heapsort implementation with the reference's exact float32 depth-bin expression,
descending integer layer, and original emission ordinal as the final tie key.
Atomic scatter arrival order cannot affect the result. Circuit border bits
remain payload, not part of circuit identity or depth/layer ordering.

There is no fixed-size per-tile candidate list or per-pixel fragment K-buffer.
Each list is counted in int64, checked, allocated and then filled; deep pixels
use the same dynamically sized CSR and unbounded run sort. Counts and offsets
must fit the renderer's int32 interfaces. Exceeding that range raises an
explicit `OverflowError` **before** a narrowed write, not partial output or a
silently truncated list. Ordinary allocation failures remain explicit or use
the existing renderer's memory retry mechanism.

This first migration still retains the coverage-window raw/sheet result and
uses allocator-owned bin/sort scratch, like the reference sort workspace. It
is not a claim that the complete renderer now fits into one tile's storage.
Raw-fragment lifetime changes from Rank 1 and direct tile-to-sheet emission
for general pixels are separate work.

## Rank 4: prove whole pixels before general sheet construction

`compact_interior_sheets` examines retained raw records **before** calling the
general compactor. A pixel is simple only when **all** of its layers pass:

1. Every record is a triangle with exactly unit analytic area, every ownership
   sample, no sliver flag, and a strict full-footprint geometry certificate.
2. Every neighboring layer has nonoverlapping distance intervals in the
   existing depth/layer walk order. Coplanarity and within-pixel crossings fail.
3. All source surface IDs are distinct. Repeated surface/facing/band cases
   stay on the general path, preserving closed-shell, fold, rank and sibling
   rules. Closed-shell flags and uncertain opacity also fail the shortcut;
   missing declarations are treated as uncertainty. Custom pipelines fall back.

Distinct transparent layers are **all kept**. This is an ordered alpha stack,
not nearest-surface rendering. A per-pixel surface-ID sort checks uniqueness
without a fixed-size local set or a quadratic deep-stack scan.

Whole exceptional pixel runs are gathered into the original `compact_sheets`.
Simple runs bypass its grouping, area accumulation, rank/sibling construction,
representative selection and sample-depth ownership passes. A merge restores
the original covered-pixel order and CSR, combining ordinary sheets with direct
one-fragment sheets. No general pixel is partially processed by both routes.
All-simple batches do not call the general compactor at all.

Bit 28 (`_AA_SIMPLE_INTERIOR_BIT`) in the **sheet** mask certifies the whole
pixel. Raw masks remain unchanged. Certified sheets have unit compositing
coverage and retain the original shading reference, barycentrics, exact hit
key, and mesh-cap data. General sheet data is copied unchanged.

The shared sheet resolver uses one scalar transmittance for a certified pixel,
skipping coverage redistribution and sample-ownership work. Material evaluation,
shadow mode 1/event construction/mode 2, reflection/refraction and continuation
allocation remain shared. Diagnostic lane dumps deliberately use the equivalent
eight-lane path so they expose real lane values, not stale scalar-path state.
Scalar versus lane-sum arithmetic is subject to the renderer's existing floating
point tolerance, not a new bitwise-output guarantee.

### Boundary and crossing fallback

All partial boundaries, circuits, slivers, same-surface folds, depth crossings,
uncertain/custom materials and failed numeric proofs retain the **existing
analytic sheet path**. This does not introduce a new exact continuous polygon-
arrangement integrator or claim the current fallback exactly integrates every
arbitrary overlapping procedural-shader configuration. Keeping that consumer
is intentional in the audit's migration strategy.

The existing analytic approximations, contribution thresholds, sheet conflict
rank ceiling and ray `max_surfaces_per_ray` ceiling are unchanged. In particular,
an unbounded new candidate/ordering queue must not be described as removing
those downstream transport limits. No new approximate early exit or fixed
layer cap is added by these settings.

## Diagnostics and correctness tests

Coverage results expose `raster_tile_binning`, `raster_simple_interiors`,
`num_simple_pixels`, and `num_general_pixels`. The tiled frontend also exposes
`tile_candidates`, `tile_bbox_rejected`, and `tile_occluded`; the latter counts
triangle/tile incidences, not pixels or a promised number of saved fragments.

**Neither switch certifies anything on every scene, and a neutral timing does
not say which.** On the repository's anchor benchmark (`nn_scene_UHD`'s scene)
`tile_occluded` and `num_simple_pixels` are both exactly zero: nearly every
triangle there is a closed shell, and `_rect_proof`'s distance interval is too
loose to close on geometry that small. Read those counters beside any
measurement of these switches, and see
`benchmarks/performance/reports/t4_2026_09/tiled_primary_ab_1.md`.

```bash
python -m pytest -q tests/unit_tests/test_tiled_primary_taichi.py
python -m pytest -q tests/unit_tests/test_tiled_primary_render.py
```

The feature tests cover containment versus sample-only evidence, perspective
bounds against an independent float64 oracle, numeric uncertainty, exact wide-ID
ordering and ties, dynamic deep lists, nonzero multi-frame windows, partial
coarse/fine tiles, pre-emission opaque rejection, raw/sheet record parity,
whole-pixel fallback, and actual bypass of general metadata construction.
Render tests exercise both switches independently and together, repeated
renders, transparent stacks, circuits, shadows, and reflective/transmitting
materials. They compare pre-encoding frames under the existing channel tolerance.

No performance measurement or default promotion accompanies this implementation.
