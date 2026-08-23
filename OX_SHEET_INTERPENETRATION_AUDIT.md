# Audit: how the sheet resolve orders INTERPENETRATING surfaces

Read-only source audit. Basis: `CLAUDE.md`,
`algan/rendering/raytracing/DESIGN_sheet_resolve.md`,
`algan/rendering/raytracing/{raster_taichi,sheets,sheet_compact_taichi,sheet_resolve_taichi,raster_pipeline,tracer,wavefront_kernels_taichi}.py`,
`algan/rendering/raytracing/raytrace_kernels_taichi.py`, and
`tests/full_renders/scenes/solids_and_camera.py`. No files were modified; no
renders, no tests. Where the design document and the code disagree, the code is
stated as authoritative and the disagreement is called out (two cases below:
§6.1/Q4, and none other found).

## Summary (10 lines)

**Verdict: YES — the hypothesis holds as stated.**
1. A fragment carries ONE scalar depth: the exact camera distance evaluated at
   the centroid of the sub-pixel samples it owns (raster_taichi.py:1334–1361),
   packed as raw f32 bits into `frag_key`'s low 32 bits (raster_taichi.py:2010–2011).
2. A SHEET carries one depth: its **nearest fragment's** scalar, recovered in
   the resolve as `t_hit = _frag_t(sheet_key[idx])` (sheet_resolve_taichi.py:265;
   built at sheets.py:943–948, 1030). The design's "depth = min" (P2,
   DESIGN_sheet_resolve.md:180) matches the code.
3. Front-to-back order is fixed ONCE on the host: sheets are sorted by the
   nearest fragment's position in the emission order — depth **bins** of width
   `DEPTH_TIE_EPSILON = 1e-4`, ties broken by descending primitive-layer index
   (sheets.py:1011, 950–951; raster_pipeline.py:1063–1093; raytrace_kernels_taichi.py:101).
4. The resolve kernel walks that array order verbatim (`while q < total`,
   sheet_resolve_taichi.py:263–264) and contains **no depth comparison between
   two sheets anywhere** — the only depth uses are the far-clip break (line 275),
   the shading position (line 371), and continuation origins.
5. The N-bit sample mask is coverage-only: it selects which samples a sheet
   dims (`slots`, lines 309–321) and normalizes the claim (`cfac`), but it never
   arbitrates which of two OVERLAPPING sheets is in front at a given sample.
   There is no per-sample depth in the pipeline at all — not even per fragment.
6. Nothing bands or splits across meshes: the group key is `(surface id,
   facing)` (sheets.py:816), so two meshes are always separate sheets, and two
   sheets of different meshes with overlapping depth ranges are ordered WHOLLY
   by fact 3's single scalar comparison.
7. Consequently, at a pixel where two surfaces cross inside the pixel, the
   entire pixel's samples are awarded sheet-by-sheet in nearest-fragment order —
   whichever surface's sheet carries the smaller representative depth claims its
   owned samples at full strength, including samples where it is actually behind.
8. The classic supersampled wavefront resolves the same pixel correctly because
   each sub-pixel sample is its own camera ray whose hits are depth-ordered per
   ray by real intersection tests (tracer.py:1915–1918;
   raytrace_kernels_taichi.py:2114–2148).
9. This explains all four measured facts: finer dice cannot help (the surfaces
   stay continuous, so banding still fuses them and the order is still
   nearest-fragment); route-off fixes it (per-ray depth); one-mesh and
   shade-split don't touch it (they act within/across same-mesh sheets);
   truncation counters stay zero (no ceiling is involved).
10. **What decides the flip, in one sentence:** the winner is decided solely by
    which surface's sheet owns the smaller *nearest-fragment* depth-bin — a
    whole-pixel, one-scalar decision made during host compaction, with the
    sample mask describing coverage only and no per-sample depth existing
    downstream of emission.

---

## 1. Trace of a fragment's depth from emission to the resolve

**Emission — one scalar per fragment, not per sample.**

Triangles: `_ss_pixel` computes the perspective-correct hit and then
re-evaluates it *at the centroid of the owned samples*
(raster_taichi.py:1308–1352):

> "Re-evaluate the fragment AT THE CENTROID OF THE SAMPLES IT OWNS rather than
> at the pixel centre … The owned samples are inside the triangle by
> construction and a triangle is convex, so their centroid is too: no
> extrapolation" (raster_taichi.py:1308–1330)

and returns the exact distance (raster_taichi.py:1353–1361):

```python
hp = b0 * v0 + b1 * v1 + b2 * v2
tt = (hp - cam_o).norm()
...
t = tt
```

Circuits: `_bez_pixel_hit` returns the plane-intersection distance `th`
(raster_taichi.py:1631–1632) — likewise one scalar.

The write kernels pack it as **raw IEEE bits into the low 32 bits of
`frag_key`**, pixel in the high 32 (raster_tri_write, raster_taichi.py:2010–2011;
identically `raster_bez_write`, lines 2133–2134):

```python
tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
```

So per sample there is no depth anywhere — the mask says *which* samples, the
scalar says *where* (one point per fragment). The mask/flag layout is
documented at raster_taichi.py:111–142 (`_AA_FLAG_SHIFT = 16`; bits 0..N−1
samples, then backface/sliver/one-mesh flag bits).

**Host ordering — the emission stream.**
`prepare_sparse_raster_coverage` sorts fragments once via
`_exact_fragment_order` (raster_pipeline.py:1552, defined 1063–1093):
descending layer first, then ascending `(pixel << 32) | floor(t /
DEPTH_TIE_EPSILON)` (lines 1073, 1082–1087), with
`DEPTH_TIE_EPSILON = 1e-4` (raytrace_kernels_taichi.py:101). The opaque-prefix
truncation (raster_pipeline.py:1602–1643) then cuts each pixel's stream after
its first proven-opaque hit.

**Compaction — the SHEET's depth is its nearest fragment's.**
`compact_sheets` re-sorts by `(pixel, group, t)` (sheets.py:823,
`order = _lexsort(pix, gkey, t)`), forms bands, then reduces each band to:

```python
first_sorted.scatter_reduce_(0, band_id, positions, reduce="amin", ...)
nearest_orig = pos_o.index_select(0, first_sorted)      # sheets.py:944–948
```

`positions` runs over the depth-sorted stream, so the `amin` picks the band's
minimum-depth fragment (bands share `(pixel, group)`, so sorted position order
is depth order). The emitted record (sheets.py:1030):

```python
sheet_key = frag_key.index_select(0, nearest_orig).index_select(0, final)
```

i.e. the sheet's key **is** its nearest fragment's key — pixel plus raw f32
depth bits. The reduce is therefore a **min**: the nearest fragment's exact
distance, matching the design's P2 "depth = min" (DESIGN_sheet_resolve.md:180)
and §1's "a representative depth" (line 71).

**Resolve — decode, never re-compare.**
`sheet_resolve_shade` recovers it per sheet
(sheet_resolve_taichi.py:265):

```python
t_hit = _frag_t(sheet_key[idx])
```

(`_frag_t`, raster_taichi.py:826–829, bit-casts the low 32 bits back to f32.)
`t_hit` positions the shading (`surf_pos = ro + t_hit * rd`, line 371) and
seeds continuations (`base_dist + t_hit`); it is never compared against
another sheet's `t_hit`.

## 2. What decides front-to-back order, and at what granularity

**Decision: the array order produced by the host; granularity: one scalar
comparison between the two sheets' nearest fragments, quantized to 1e-4 depth
bins, ties broken by arbitrary primitive-layer index.**

The final sheet order (sheets.py:1010–1011):

```python
# ---- Final order: (pixel, classic order of nearest fragment) -----------
final = torch.argsort(min_pos, stable=True)
```

where `min_pos` is the minimum **original-stream** position among the band's
fragments (sheets.py:950–951) and the original stream is the emission's
`(pixel, depth-bin, descending-layer)` order (raster_pipeline.py:1063–1093).
The CSR preserves it (`sheet_offsets`, sheets.py:1052–1059), and the kernel
consumes segments verbatim (sheet_resolve_taichi.py:194–196, 263–264):

```python
while q < total and processed < MAX_SURFACES_PER_RAY:
    idx = start + q
    t_hit = _frag_t(sheet_key[idx])
```

**There is no per-sample depth comparison between two sheets of different
surfaces anywhere in the resolve.** Every use of `t_hit` in
`sheet_resolve_shade` is: the far-clip break (`base_dist + t_hit > far_clip`,
line 275), the shading point (line 371), continuation spawn bases (lines 643,
655, 693, 756, 786, 815, 856, 866, 913, 925), the bounce bookkeeping
(`base_dist += t_hit`, lines 704, 924), and dump rows. None compares it with
another sheet's depth; the occlusion write itself is depth-blind (see Q3).
A grep of the whole sheet pipeline for cross-sheet depth logic finds none.

Two consequences worth stating plainly:

* Because order keys on the nearest fragment's **bin**, two sheets whose
  nearest fragments fall inside one 1e-4 bin are ordered by **descending layer
  index** (raster_pipeline.py:1073) — i.e., by merge-order primitive numbering,
  effectively arbitrary with respect to geometry. Exact depths closer than the
  bin width do not participate in the decision at all.
* The decoded `t_hit` used downstream is the exact f32, so a sheet's shading
  position and its walk position come from slightly different quantizations of
  the same nearest fragment. Harmless except as confusion; noted for accuracy.

## 3. The sample mask: coverage only, not occlusion

**N = 8.** `_AA_NUM_SAMPLES = len(_AA_SAMPLES)` (raster_taichi.py:213) with
`_AA_SAMPLES` built from `_AA_PATTERN_8` (lines 202–212) — the D3D-standard
sparse 8-sample pattern, fixed-point 1/4096-pixel lattice units
(lines 187–190). (`_AA_PATTERN_16` exists, lines 205–208, but is unused by the
tuple comprehension; switching requires editing that line and clearing the
kernel cache, per the comment at 195–199.)

**Positions:** raster_taichi.py:202–204 — e.g. `(1, −3), (−1, 3), (5, 1), …`
in 16ths of a pixel from the centre; consumed by the ownership test at
lines 1077–1088.

**Role in the composite — coverage only.** In `sheet_resolve_shade`:

* `slots` = which samples the sheet owns (lines 309–321):

```python
slots = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
...
for s in ti.static(range(_AA_NUM_SAMPLES)):
    if ((msk_low >> s) & 1) == 0:
        slots[s] = 0.0
```

* `svis` = per-sample transmittance, initialized to 1 (line 243) and multiplied
  by `_run_svis_write` (raster_taichi.py:769–793), which is exactly §4.3's
  `T[s] *= (1 − a[s]) + a[s]·ts`:

```python
ak = cfac * a_s * slots[s]
fct = (1.0 - ak) + ak * trans_share
...
svis[s] *= fct                                   # raster_taichi.py:784–792
```

The mask decides **which** samples a sheet dims and how strongly — it is never
compared against another sheet's depth to decide **whether** it may dim them.
Occlusion between sheets is purely walk order: a sheet attenuates the samples
its mask owns regardless of whether a later-walked sheet is actually nearer at
those samples. (Within one mesh, near/far sheets still order correctly because
their nearest fragments inherit true depth order; the failure mode needs two
surfaces.)

**§4.3 arithmetic mapped onto the code** (terms: `Q`, `corr`, `a[s]`,
`weight`, `T[s]`; kernel column, oracle column):

| Design term | Resolve kernel (`sheet_resolve_taichi.py`) | Oracle (`sheets.resolve_pixel_reference`) |
| --- | --- | --- |
| `Q = popcount(mask)/N` | `pop = _popcount_samples(msk_low)` (313) — Q enters only through the denominator below | `bin(msk_low).count("1")/N` (190) |
| `corr = min(area,1)/max(Q, 1/N)` | `cfac`: full-union branch 315–317 (`corr=1` inside `_AA_FULL_DUST`, else `area`), partial branch 319–323 `cfac = area*N/pop` | lines 185–190 (`areal` 186, full-union dust 188, partial 190) |
| `a[s] = alpha·slot[s]·corr` | `ak = cfac * a_s * slots[s]` (raster_taichi.py:784), with `a_s = mat_alpha*dens` (resolve 484), `dens = area` for areal sheets (310–311) else 1, `alpha = clamp(mat_alpha*eff)` (482–483) | `c = [p_i*own[s]]` (192) → `a = [alpha*w*own[s]]` (222) |
| `weight = Σ_s T[s]·a[s]/N` | `vis += slots[s]*svis[s]` (324–326); `eff = vis*_AA_SAMPLE_WEIGHT*dens*cfac` (327); committed as `acc += weight·alpha·color` (568–571) | `eff = sum(T[s]*c[s] for s)/N` (193); `claims.append(alpha*eff)` (212) |
| `T[s] *= (1−a[s]) + a[s]·ts` | `_run_svis_write(svis, slots, w_a_s, trans_share, w_cfac, 1)` (calls at 768, 817, 872, 939; body raster_taichi.py:783–792) | `T[s] *= fct` (225–230) |

(The extra `dens` factor and the `w_cfac/w_a_s` band indirection are §4.4's
sibling arithmetic — negative `sheet_cov` marks a continuing band, resolved at
sheet_resolve_taichi.py:270–271, 341, 488–493 — orthogonal to this audit.)

## 4. Interpenetration: every place that handles — or declines to handle — it

1. **The sheet key never crosses meshes.** `gkey = torch.where(is_tri, sid * 2
   + facing, -(positions + 2))` (sheets.py:816); a band starts at any
   pixel-or-group change (`new_group[1:] = (pix_o[1:] != pix_o[:-1]) |
   (g_o[1:] != g_o[:-1])`, sheets.py:829–831). **Nothing bands or splits across
   meshes**: two meshes are always two groups, hence always separate
   bands/sheets, whatever their depth relation.

2. **Two sheets of different meshes whose depth ranges overlap** are ordered by
   the single mechanism of Q2 — `final = torch.argsort(min_pos, stable=True)`
   (sheets.py:1011) — i.e., by whichever sheet's *nearest* fragment sits in the
   earlier depth bin (ties: descending layer). Their overlapping extents are
   invisible to the decision; the later sheet simply finds dimmed `svis` on
   whatever samples the earlier sheet claimed. That is the entire code path.

3. **The fusion detector and conflict rank are same-surface machinery.**
   `sheet_fused` exists because "a band in which any sample bit was contributed
   twice has provably fused at least two sheets" (sheets.py:52–55) — of *one*
   `(mesh, facing)`; the rank splits "overlapping translucent layers of one
   mesh" (sheets.py:863–866) and its ceiling is documented as "overlapping
   layers of **one surface**" (sheets.py:84–87, truncation.py:22–25). Cross-mesh
   overlap gets no analogous treatment — by construction it never fuses, so
   nothing detects or resolves it.

4. **The one explicit cross-surface acknowledgment declines to act.**
   `_sibling_weights` (sheets.py:645–649):

   > "where another surface interleaves them (a coincident depth) the band
   > closes early and its remainder composites sheet by sheet -- the pre-split
   > behaviour, on a pixel where the depth order was **already ambiguous**."

   Mirrored in DESIGN_sheet_resolve.md:305–311. "Already ambiguous" is the
   design conceding that interleaved-depth pixels are not resolved — merely
   handled consistently.

5. **The opaque-prefix truncation is depth-blind across meshes too.** Each
   pixel keeps the prefix through its *first proven-opaque hit*, whichever mesh
   it belongs to (raster_pipeline.py:1602–1624, `first_opaque` scatter-amin
   over `opaque_pos`). It participates in cross-surface outcomes only insofar
   as it can delete a farther surface's fragments outright once a nearer
   full-coverage opaque fragment is kept — the ordinary case works because
   "nearer" is usually correct; a mis-ordered crossing pixel is cut just the
   same. (Not implicated by the measured facts — counters zero, dice-invariant
   — but it is part of the ordering machinery.)

6. **The resolve kernel itself has no notion of it.** `sheet_resolve_shade`
   contains no comment, branch, or data path mentioning two surfaces crossing;
   its occlusion write (`_run_svis_write`) takes only `(svis, slots, a_s,
   trans_share, cfac)` — no depths at all (raster_taichi.py:770).

7. **Where the design document oversells the code.** §6.1 states
   "Inter-sheet overlap is sampled at N positions"
   (DESIGN_sheet_resolve.md:405–409), and raster_taichi.py's facing-bit comment
   still says "each sample composites in true depth order"
   (raster_taichi.py:123–126 — written for the deleted fragment walk). Under
   the sheet resolve, per-sample transmittance exists, but the *order* it
   accumulates in was fixed per sheet, per pixel, by one scalar — so overlap
   between sheets of different surfaces is **not** sampled-resolution in any
   sense that involves depth at the sample positions. Both fragments of those
   comments predate or describe the fragment walk; the code wins, per the
   brief's standing instruction.

*(Reconstruction of the observed Act-3 flip from these facts — **inferred,
not read**: the shaft pierces the sphere continuously, so the shaft's
front-facing fragments inside one straddling pixel form ONE band (no depth gap
— banding splits only on gaps exceeding twice the pair's own per-pixel scale,
sheets.py:505–569, invoked at 835–851 with `band_rule="prim", band_c=2.0`,
raster_pipeline.py:1791–1792) and ONE sheet whose nearest fragment is the
exit-side (camera-nearer) part. That sheet therefore walks ahead of the
sphere-front sheet and claims every sample its mask union covers — including
samples where only the buried portion projects — at full material strength,
before the sphere sheet can claim them. Dice refinement shrinks fragments but
cannot open a depth gap on a continuous surface, matching measurement; route
off restores per-ray depth, matching measurement. Which samples flip, and why
some pixels go fully green, follows from the union/corr bookkeeping above and
varies with sub-pixel phase — consistent with a 1–2 px band of isolated
full flips.)*

## 5. What the classic supersampled wavefront does differently

With the route off, `analytic_raster_route_active` returns False
(tracer.py:542–635; the veto list at 567–581 names exactly the toggles the
brief lists), `aa = max(1, anti_alias_level)` (tracer.py:1208), and the classic
gen → traverse → shade → composite pipeline runs **once per sub-pixel sample**:
`for si in range(aa): for sj in range(aa): jx = (si + 0.5) * inv_aa; jy = ...`
(tracer.py:1915–1918), each pass generating one primary ray at that jittered
position (`wavefront_generate_rays`, wavefront_kernels_taichi.py:1704–1740).
Each such ray independently traverses the BVHs and gathers its KBUF nearest
hits by **actual intersection** (`_collect_hits`, raytrace_kernels_taichi.py:1989+;
`hit_ok, w1, w2, t = _tri_hit(ro, rd, v0, v1, v2)`, line 2114), keeping hits
ordered by the strict per-ray comparator `_comes_after`
(raytrace_kernels_taichi.py:862–880) — distance-binned with layer tie-break,
but per RAY, so two surfaces crossing inside a pixel are separated wherever
any sample's ray sees them in different order. **The place it resolves depth
per sample is the hit-accept/keep-nearest block in `_collect_hits`
(raytrace_kernels_taichi.py:2114–2148)** — `accept = ... _comes_after(t, layer,
t_prev, layer_prev) ...` driving insertion into the per-ray K-buffer — fed by
per-sample rays generated at tracer.py:1915–1918 /
wavefront_kernels_taichi.py:1737–1740. A pixel is thus the average of `aa²`
independent, correctly-ordered resolves; the sheet route replaces those `aa²`
depth decisions with ONE per pixel (Q2), which is precisely the regression
surface.

## 6. Where a fix would have to go (scoped, not written)

First, the datum question, asked head-on: **does any sheet field describe its
depth EXTENT within the pixel? No.** The sheet record is exactly
(`raster_pipeline.py:1818–1830`, produced at sheets.py:1033–1050): `sheet_key`
(pixel ‖ nearest-fragment f32 depth bits), `sheet_ref`, `sheet_ab`,
`sheet_cov`/`sheet_wgt` (areas), `sheet_msk`/`sheet_wmsk` (masks+flags),
`sheet_cap` (an AREA ceiling, not depth), `sheet_nfrag`, `sheet_fused`,
`sheet_offsets`. One depth, no max, no slope, no footprint. The closest
existing quantity — the per-triangle camera-distance extent and its per-pixel
slope — is computed transiently by `_prim_split_after` (sheets.py:536–541
`ext = dmax - dmin`; 548–560 `slope = ext / proj`) and **discarded** after the
band-gap test; it never reaches a sheet record. Any per-sample tie-break will
most plausibly build on reviving exactly that computation as persistent data.

Functions that would need to change, with the data each lacks:

1. **`sheets.compact_sheets`** (sheets.py:695). Would have to emit, per band,
   a second depth datum — at minimum the farthest fragment's t (symmetric to
   the existing `amin` at 944–947; a `scatter_reduce_("amax")` beside it), or a
   per-band depth slope/extent. Lacks: any farthest-depth/extent reduction and
   the output fields to carry it.
2. **`sheets._band_reduce` + `sheet_compact_taichi.sheet_band_reduce`**
   (sheets.py:351; sheet_compact_taichi.py:121–159). The natural home for
   min/max-depth reduction alongside area/union/dup. Lacks: the depth array is
   not even an argument — the kernel signature receives only
   `(band, msk, cov, n, mask_all, sliver_bit, ...)` (sheet_compact_taichi.py:122–134);
   the sorted depth stream (`t_o`) would have to be threaded in, and the f64
   determinism discipline of §10.4 applied to whatever float extent is reduced.
3. **`raster_pipeline.prepare_sparse_raster_coverage`, compaction hand-off**
   (raster_pipeline.py:1796–1830). New persistent arena arrays for the extent
   field(s) plus the copy block; the resolve's 64-runtime-argument ceiling
   (noted at sheet_resolve_taichi.py:225–228) means new per-sheet inputs likely
   ride existing arrays or `layer_offsets`-style packing rather than fresh
   parameters.
4. **`sheet_resolve_shade`** (sheet_resolve_taichi.py:106). The actual
   tie-break: before committing a sheet's claim on sample s, compare the
   current sheet's depth range against the *next* sheet claiming that sample
   (requires either look-ahead state in the walk or a pre-pass). Lacks: any
   per-sample notion of which surface is nearer — it has one scalar `t_hit`
   (265) and would need the following sheet's extent, or a per-sample depth
   estimate (e.g. plane/gradient evaluation at each of the N sample positions,
   which neither the sheet nor the fragment record carries — fragments store
   one centroid depth, Q1).
5. **`raster_taichi._run_svis_write`** (raster_taichi.py:769–793). The write
   factor would need per-sample depth arguments (or a caller-computed per-sample
   gate replacing `slots[s]` with a 0/1 visibility decided by depth, not
   membership). Lacks: everything depth-related, by signature.
6. **`sheets.resolve_pixel_reference`** (sheets.py:107–238). The §2 oracle must
   implement the identical rule or parity harnesses (`_aa_dump_check` lineage,
   the Phase-2 8.94e-08 pin) break. Lacks: same per-sheet extent inputs.
7. **Constraint to respect:** the mode-1 shadow-event build shares this kernel
   body (sheet_resolve_taichi.py:138–147, 432–480); the file's stated invariant
   is that the event pass and the shading pass agree about transport, so any
   per-sample reorder must be visible identically in both modes, and §4.4's
   sibling bands (negative-coverage deferral, 254–256, 488–493) must not be
   reordered mid-band — the design already documents that pulling a sibling
   forward past another surface's sheet flips which face paints the pixel
   (DESIGN_sheet_resolve.md:305–311).

A minimal fix shape consistent with the above (for scoping only): persist
per-band farthest-t (function 2), emit it as `sheet_far` (functions 1, 3), and
in the resolve defer each sample's attenuation until a sheet is known to be
frontmost *at that sample* under `(t_near, t_far)` interval tests against the
remaining sheets claiming it (functions 4–6) — falling back to walk order when
intervals nest ambiguously, exactly as §4.4 already falls back for interleaved
siblings.

---

### Verification trail

Steps taken: read `CLAUDE.md`; read `DESIGN_sheet_resolve.md` in full (882
lines); read `sheets.py` (1060), `sheet_resolve_taichi.py` (1081),
`sheet_compact_taichi.py` (221) in full; read the emission/order/truncation/
compaction sections of `raster_pipeline.py` (1040–1871) and the constants,
coverage, and write-kernel sections of `raster_taichi.py` (60–2209); read
`analytic_raster_route_active` and the AA loop in `tracer.py` (530–659,
1180–1299, 1860–1979, 2410–2499); read `wavefront_generate_rays`/
`wavefront_traverse` heads (wavefront_kernels_taichi.py:1700–1849) and
`_collect_hits`/`_comes_after` (raytrace_kernels_taichi.py:840–880, 1989–2183);
grepped the raytracing tree for every occurrence of
interpenetration/overlap/crossing language (§4's list is exhaustive for the
sheet pipeline); confirmed the Act-3 scene geometry
(tests/full_renders/scenes/solids_and_camera.py:169–189: three
`thickness=0.05` arrows from `ORIGIN`, `Dot3D(radius=0.14)` at the origin,
`Line3D(thickness=0.03)`). No steps remain; the report is complete.
