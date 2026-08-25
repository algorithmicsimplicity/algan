# Batch-window dependence audit: which per-batch decisions change rendered pixels?

Read-only audit of `scratch_perf/ox/brief_batchwide_audit.md`. No tracked file was
modified; probes ran host-side on CPU tensors
(`scratch_perf/ox/probe_batchwide_audit.py`, run with
`ALGAN_USE_DAEMON=0 ALGAN_RENDER_DEVICE=cpu uv run python ...`). Nothing was rendered.

Measured premise being explained: same scene, 3-frame vs 19-frame batches differ at
3840x2160 (~5% of pixels: glyph edges, diced-sphere silhouettes, speckle on textured
quads) and are byte-identical at 704x396; renders with identical batch windows are
byte-identical regardless of chunk/tile split.

## Summary table

| # | Mechanism | Site | Verdict | Class | Coupled frames | Monotone under longer window? | Visible pop at a batch boundary? |
|---|---|---|---|---|---|---|---|
| 1 | Bezier chord count chosen once per segment from max error over **all frames of the batch** | `primitives.py:2873-3031`; kernel `logical_pn_taichi.py:297-405` | **CONFIRMED (b)** | tessellation values | all frames of the window, via a max | Yes — finer only (max error can only grow) | Yes — silhouette/stroke geometry changes across windows (matches the measured glyph-edge diffs); mid-render pop if the OOM retry shrinks the window |
| 2 | Logical PN edge/patch dice levels | `primitives.py:1556-1619, 1741-1800, 1894-1986` | **REFUTED** (per-frame by construction) | — | none | — | — |
| 2a | PN budget `max_diced_triangles` (`_triangle_counts(...).sum(1)`) | `primitives.py:1794, 1977` | REFUTED — `sum(1)` is over **patches**, giving a per-frame total `[T]` | frame-local by explicit design (`:1308-1320`) | none | — | — |
| 2b | Static-mesh collapse + patch dedup (`_collapse_redundant_frames`, `share_patches`) | `primitives.py:2100-2129, 1599, 2384` | REFUTED as a value coupling | (c) layout-only, fires only when frames are byte-equal | condition spans the window, result value-preserving | n/a | no |
| 2c | Diced width = widest frame of the batch (`counts.sum(1).amax()`) | `primitives.py:2208-2210` | CONFIRMED | (c) layout/padding — feeds the documented STBVH-order → depth-tie mechanism | all frames via a max | not monotone (depends which frames are in the window) | only through tie/speckle machinery |
| 3 | Constant-property promotion judged over the window's frames (`.all(0)`) | `scene_builder.py:536-636` (`:566, :575, :576`) | **CONFIRMED (b)** | shading-path choice + primitive reordering | every frame of the window (predicate: corner-uniform in *every* frame) | Reverse-monotone: a longer window can only **shrink** the promoted set | Yes — at any step change crossing a boundary; plus ulp-level arithmetic difference between texel and vertex paths |
| 3a | `_dedup_time` / `MERGE_DEDUP_TIME` | `scene_builder.py:525-533, 1639-1648, 1829-1836` | CONFIRMED firing window-dependently | (c) layout-only — consumers read `f % shape[0]`; collapses only byte-equal frames | all frames (equality predicate) | reverse-monotone | no (values unchanged) |
| 3b | `_unify_time` broadcasting | `raytracing/utils.py:37-48` | REFUTED | (a) stride-0 expand, values unchanged | none | n/a | no |
| 4 | STBVH structure depends on frame count/window | `stbvh.py:503-607, 792-793, 819-820`; `refit_bvh.py:309-319` | **CONFIRMED** | tree shape/order; pixel-visible only via epsilon/tie sites below | all frames (dyadic segmentation, quantization bounds, SAH union boxes) | not monotone (dyadic interval boundaries move arbitrarily) | yes, at the epsilon level |
| 4a | Where a different tree changes a hit: depth-tie bins broken by merged-column index | `raytrace_kernels_taichi.py:976-991, 1140`; windows `:1074-1089`; `sheets.py:1444-1448`; `raster_pipeline.py:1085` | CONFIRMED | (b) via merge order (layer = prim index) | — | — | coplanar/near-coplanar pixels flip |
| 4b | Shadow-march seam de-dup within `DEPTH_TIE_EPSILON`; f16 box rounding admits ulp-level candidates | `raytrace_kernels_taichi.py:2940-2945` (also `:3168, :3353, :3524, :3674`), `wavefront_kernels_taichi.py:2624`; `stbvh.py:84-96` | CONFIRMED | (b) epsilon-level | — | — | sub-pixel speckle |
| 5 | Surface weld flags / texture closed-axes reduced over ALL dims incl. time | `surface.py:344-368, 371-400`; called per batch `:3012-3032, 703-719`; wrap-pad `:3101-3113, 3127-3134` | **CONFIRMED (b)** (conditional) | topology (triangle count) / texture sampling | all frames of the grid materialization (tolerance 1e-4 predicate) | reverse-monotone (longer window can only unweld/unpad) | Yes — triangle-count/texture-seam pops when closure state crosses the tolerance inside a window |
| 5a | `compute_grid_vertex_normals` | `surface.py:525-700` | REFUTED | (a) all reductions spatial/per-frame | none | — | — |
| 6 | Bezier endpoint-vertex OR over frames | `primitives.py:3056-3072` | CONFIRMED | (c) packed vertex COUNT (layout) | all frames via `.any(0)` | monotone (longer can only add) | only via merge-width → tie landing |
| 6a | Circuit AABB inflation `amax(0)` | `primitives.py:3374-3383` | CONFIRMED | (c) BVH bounds only | window via max | not monotone | no direct pixel effect |
| 6b | Per-render conservative folds (`shadow_cast_flag`, `closed_shell_ceiling_flag`, routing flags `has_refractive`/`has_transmissive`/`tri_has_reflective`/`bez_*`, mem-trim bands, `_decode_material_block_colors`) | `primitives.py:347-389`; `scene_builder.py:963-967, 1063, 1158-1171, 1703-1713, 1733-1807, 1841-1843, 1878-1880, 2051` | REFUTED as pixel couplings | (c)/(a): inputs are per-mob constants today, or the flag only enables a capability whose per-pixel answer is unchanged | window via or/amax | — | no |

Bottom line: three mechanisms make **pixel values** depend on which frames share a
batch — (1) bezier chord counts, (3) constant-property promotion, (5) weld/wrap-pad
topology — plus the family of epsilon/tie effects reached through batch-dependent
layout (2c, 4, 4a, 4b, 6). Mechanism 1 is the dominant explanation for the measured
glyph-edge and silhouette differences; 4a/4b explain resolution-sensitive speckle;
3 and 5 add step-change sensitivity.

---

## Q1 — Bezier chord subdivision: CONFIRMED (b)

`RayTracedBezierCircuitPrimitive._compute_samples_per_segment`
(`algan/rendering/raytracing/primitives.py:2873-3031`) chooses **one chord count per
cubic segment for the whole render batch**: *"We retain the first level whose bound is
no larger than `num_pixels_per_sample` for every frame in the render batch"*
(`:2884-2886`).

- Torch path: per-level error is reduced over frames and subcurves,
  `frame_error_squared = error_squared.amax(dim=(0, 2))` (`:3018`), accumulated across
  frame chunks with `torch.maximum` (`:3019-3021`; the chunking comment at `:2956-2963`
  notes the reduction is a pure max, so chunk size cannot change the result).
- Fused-kernel path: `bezier_chord_hull_error`
  (`algan/rendering/raytracing/logical_pn_taichi.py:297-405`) runs one thread per
  (segment, frame, subcurve) — `per_segment = num_frames * num_subdivisions` (`:326`) —
  and folds each thread's error into the segment's slot with
  `ti.atomic_max(error_squared[a], result)` (`:405`).
- The winning level ladder ascends powers of two (`:2924-2952`), so the count is the
  first level meeting tolerance under the **max** over the window's frames.

Probe: one synthetic segment whose screen-space bow grows over 19 frames resolves to
**128** chords over the full window vs **16/32/64/128** per 3-frame sub-window —
every short-window count ≤ the long-window count (monotone).

- Decision: sample/chord count per segment.
- Couples: all frames of the batch (max-reduction).
- Monotone: **finer only** — adding frames can only raise the max error and hence the
  count; a longer batch never coarsens. (The camera-plane guard reporting `inf`
  `:3006-3017` / `logical_pn_taichi.py:386-387` is likewise monotone.)
- Pop: the polyline silhouette of every glyph/stroke depends on the window, so two
  renders of the same animation with different batch sizes draw different edges —
  exactly the observed text-glyph deltas. Within one render there is no temporal pop
  (each frame belongs to one batch), but the OOM retry shrinks the frame window
  mid-render, which re-decides counts partway through a video.

## Q2 — Logical PN dicing: REFUTED (levels are per frame)

`_required_subdivision_levels` returns per-patch interior levels `[T, P]` and per-edge
boundary levels `[T, P, 3]`, *"both of which vary freely from patch to patch and from
frame to frame"* (`primitives.py:1567-1577`).

- `_required_edge_levels`: state tensors are `[num_frames, num_patches, 3]`
  (`:1760-1769`); errors are per `(frame, patch, edge)` row — the torch fallback
  reduces `.amax(dim=(1, 2))` over chords and samples *within one curve* (`:1891`),
  and the kernel atomic-maxes within one selected row
  (`pn_edge_chord_error`, `logical_pn_taichi.py:435-493`). No reduction across the
  frame axis.
- `_required_patch_levels`: levels `[T, P]` (`:1905-1924`); the criterion kernel
  evaluates per selected (frame, patch)
  (`pn_patch_flatness_error`, `logical_pn_taichi.py:189-280`). No cross-frame
  reduction.
- **Budget axis**: `_triangle_counts(levels)` is `4 ** level` per patch
  (`:1542-1554`). In the edge search,
  `blocked = self._triangle_counts(proposed.amax(-1)).sum(1) > budget` (`:1794`):
  `proposed` is `[T, P, 3]`, `.amax(-1)` → `[T, P]` (worst edge per patch),
  `.sum(1)` sums over the **patch axis**, yielding one total **per frame** `[T]`;
  `blocked[frames]` then gates promotion per frame. Same shape in the patch search:
  `_triangle_counts(proposed).sum(1)` on `[T, P]` → `[T]` (`:1977`). This is
  deliberate — `max_diced_triangles` is documented as frame-local precisely so it
  cannot make meshes pop at batch boundaries (`:1307-1320`).
- Caveats that do **not** couple values: `_collapse_redundant_frames`
  (`:2100-2129`) drops the frame axis only when every frame is byte-equal, and the
  resulting patch dedup (`share_patches`, `:1599`; `_PatchChunk`, `:83-133`,
  `:2384`) evaluates once per distinct patch and fans out before the per-frame
  projection — value-preserving by construction (class c). The batch's widest frame
  sets the padded diced width (`counts.sum(1).amax()`, `:2208-2210`; padding
  alpha-zeroed `:2326-2328, 2453, 2463-2469`) — layout-only (class c) that feeds the
  Q4 tie machinery.

## Q3 — The merge

### `_dedup_time` / `MERGE_DEDUP_TIME` — window-dependent, value-preserving (c)

`_dedup_time` (`scene_builder.py:525-533`) collapses a leading time axis only when
`(x == x[:1]).all()`; consumers index `f % shape[0]`, so a length-1 axis reads
identical values. Applied to the triangle tables when
`SETTINGS.raytracing.MERGE_DEDUP_TIME` (`:1639-1648`) and the bezier tables
(`:1829-1836`, with the accel-table consistency note `:1822-1828`). Whether it fires
depends on the window (a table constant within each 3-frame window but varying across
a 19-frame window dedups in one and not the other), but the fired and unfired paths
hold the same per-frame values. Layout/memory only; no pixel route.

### `_unify_time` — REFUTED (a)

`raytracing/utils.py:37-48` expands time dims of 1 to the common length with
`expand` (stride-0 views later materialized by `cat`); values are replicated, never
changed. Which inputs arrive single-frame is a property of the mob's materialization,
not of the window.

### `_split_promotable` — CONFIRMED (b)

`scene_builder.py:536-636`. Per-triangle promotability:

```
color_eq = (colors == colors[:, :, :1, :]).all(-1).all(-1).all(0)   # :566
mat_eq   = (...corner pairs equal...).all(0)                        # :568-575
nonglow  = (e[..., 9:12] == 0).all(-1).all(0)                      # :576
```

Each `.all(0)` runs over the **batch's frame axis**: the requirement is
corner-uniformity *in every frame of the current window*, not equality across frames.
A triangle that is corner-uniform during frames 0-2 but carries a corner gradient from
frame 3 on is promoted in a 3-frame early window and kept per-vertex in a 19-frame
window. Probe: `kept=0 promoted=1` for T=3; `kept=1 promoted=0` for T=6 and T=19.

Does the promoted path shade identically to the per-vertex path?

- **Values**: yes. The texel is the representative's corner-0 colour/material
  `[T',5]` (`:606-618`), which equals every corner in every promotable frame; both
  routes cross the same sRGB→linear boundary — promoted maps are decoded at append
  (`_append_texture(is_color=True)`, `:1356-1359`) and per-vertex colours by
  `_decode_merged_colors` (`:1976, :1998-2035`).
- **Arithmetic**: not bit-identical. The kernel's textured branch samples a 1×1 map —
  clamped UVs give weights exactly `(1,0,0,0)`, an exact texel read
  (`_sample_texture`/`_flat_triangle_color`, `raytrace_kernels_taichi.py:1517-1596`)
  — while the per-vertex branch computes `w0*c + w1*c + w2*c`
  (`_triangle_color`, `:1494-1505`), and barycentric weights (`1-a-b`, a, b) need not
  sum to exactly 1.0, so the interpolated constant can differ from the texel by ~1
  ulp before shading. Material params: texel fetch vs per-corner `tri_extra`
  interpolation (`_triangle_extra`, `:1600-1611`) — same class of difference.
- **Layout**: promoted triangles are reordered last and grouped by value
  (`:593-599`, order applied `:1445-1460`), changing every downstream primitive index.
- Active by default on the deterministic route
  (`_constant_promotion_active`, `settings.py:2805-2813`). The legacy
  `WF_TEXTURED` route has its own three-group promoter with the same
  over-frames `const_mask` (`scene_builder.py:1063`, `_build_textured_scene`
  `:1121-1194`).

For the record: decision = which triangles render from 1×1 maps and where they sit in
the merged order; couples = every frame of the window (equality predicate);
monotone = **reverse** (longer window ⇒ promoted set can only shrink; a longer batch
can demote but never promote); pop = yes at step changes crossing a batch boundary,
plus the ulp-level path difference wherever promotion status flips.

### Other merge reductions classified

- Mem-trim band masks `.any(0)`/`.all(0)` (`:963-967`): layout permutation documented
  byte-identical (`:935-946`) — (c).
- Routing flags — `tri_has_reflective`/`has_strong_reflective` (`:1703-1714`),
  `has_refractive` (`:1733-1735`), `has_refl_transparent` (`:1761-1786`),
  `has_transmissive` (`:1796-1807`), `bez_has_nondegenerate_edges` (`:1841-1843`),
  `bez_has_reflective` (`:1878-1880`): batch-wide ORs that enable a capability only
  when the window's content demands it; the per-pixel answers on content present in
  both windows are unchanged — (c). Inputs (`no_shadow_cast`, `closed_shell`,
  `transmission` folds in `shadow_cast_flag`/`closed_shell_ceiling_flag`,
  `primitives.py:347-389`) are per-mob constants today, so their frame reductions are
  trivially window-invariant — (a) in practice, (c) by design.
- `_decode_material_block_colors` `.all(0)` (`:2051`): its input `tri_mat_id` is built
  single-frame (`torch.full((1, N), ...)`, `primitives.py:833-835`), so the reduction
  is vacuous — (a).
- `_build_textured_scene` masks (`:1158-1171`): legacy unsupported route — (c).

## Q4 — STBVH / refit BVH: CONFIRMED (structure depends on the window)

- **Instance segmentation** (`segment_primitives_in_time`, `stbvh.py:503-607`): the
  time axis is padded to a power of two (`:544-549`) — 3→4, 19→32 — and instances are
  emitted at **dyadic intervals of the batch** with a tightness test over each block
  (`:554-595`). Probe: the same 19-frame motion yields 6 instances with tspans
  `(0,15),(16,17),(18,18)` over the full window, but 4 instances
  `(0,1),(2,2)` in each 3-frame sub-window — structurally different instance sets.
- **Ordering**: Morton quantization bounds are min/max over the batch's instances
  (`:819-820`; split-builder normalization `:792-797`), so the sort keys — hence leaf
  assignment — change with the window even for identical instance sets.
- **Refit BVH** (`refit_bvh.py`): topology comes from a binned-SAH build over the
  **batch-union boxes** `ulo = vlo.amin(0)[pids]; uhi = vhi.amax(0)[pids]`
  (`:309-319`) — different window, different unions, different splits, different leaf
  permutation (`:330-418`).
- Leaf count/tree size follow the instance count M (`:768-781`).

Where a different tree/order changes a hit result (exact-arithmetic hits are
documented arrangement-invariant — `stbvh.py:80-82, 292-3008` — so these are all
epsilon/tie routes):

1. **Depth-tie bins broken by merged column index.** `_comes_after`
   (`raytrace_kernels_taichi.py:976-991`) floors distances into
   `DEPTH_TIE_EPSILON`-wide bins (`DEPTH_TIE_EPSILON = 1e-4`, `:126-128`) and orders
   same-bin hits by descending `layer`; `layer = layer_offset + prim` (`:1140`) is the
   **merged-array column**, i.e. the winner of a coplanar tie is the largest primitive
   index. Merge order shifts with the window (promotion reordering, padding widths,
   Q6's vertex counts), so tie landing moves. Its own comment records that the peel
   order formerly depended "on KBUF (and on the BVH build)" (`:980-987`). Host-side
   sheet route: same bin key (`raster_pipeline.py:1085`), same-tie conflict gate
   "strictly nearer beyond `DEPTH_TIE_EPSILON`" (`sheets.py:1444-1448`), within-bin
   order following band/fragment (merge-derived) order.
2. **Traversal windows vs conservatively-rounded boxes.** Node visits are gated to
   `[t_prev - EPS, min(best_t, t_cap) + EPS]` (`:1074-1076, :1087-1089, :1157-1161`,
   shadow march `:2917-2924`), and f16 sibling blocks round outward
   (`stbvh.py:84-96`), deliberately admitting "candidates within a float ulp of the
   acceptance boundary that the exact boxes cull — measured as epsilon-level image
   changes". Different instance boxes (different segmentation) shift which candidates
   sit on those boundaries.
3. **Bezier seam de-duplication** in the shadow march: an edge hit within
   `seam_eps = DEPTH_TIE_EPSILON` of the previous edge hit is skipped
   (`raytrace_kernels_taichi.py:2940-2945`; sibling marches `:3168, :3353, :3524,
   :3674`; wavefront `wavefront_kernels_taichi.py:2624`) — discovery-order sensitive,
   which is why the bezier tree pins the morton builder
   (`stbvh.py:300-306`: "seam de-duplication is discovery-order sensitive").
4. Any-hit shadow queries early-out on the first occluder — boolean answer is
   order-independent for opaque occluders, and transmissive batches are kept off those
   modes host-side (`:2984-2988`), so no additional route.

Consistency check: because the peel's total order is fixed given the merge layout,
renders sharing a batch window are byte-identical regardless of chunk/tile split —
matching the brief's measured fact.

## Q5 — Surface normals / welds

- `compute_grid_vertex_normals` (`algan/mobs/surfaces/surface.py:525-700`):
  **REFUTED** — every reduction is over grid axes/triangle fans; the closed-seam and
  pole masks (`is_closed_x/y`, `is_pole`, `:612-698`) reduce over spatial+coordinate
  dims only and are evaluated per frame. No frame-axis reduction.
- `get_render_primitives_batched` (`:703-719`) stacks `[N, T, W, H, 3]` and calls
  `surface_weld_flags(grids)` on the whole stack.
- **`surface_weld_flags` — CONFIRMED (b), conditional.** The three tests
  (`wrap_x`, `pole_lo`, `pole_hi`) `.all()` over **all** remaining dims of the grid —
  including the batch's time axis (`:361-367`), tolerance 1e-4 world units. The weld
  selects a cached triangulation whose **triangle count differs**
  ("welding a pole changes the triangle COUNT", `:325-326`; indices `:261-342`), and
  `get_render_primitives` runs once per render batch (`:3012-3032`). Probe: a u-seam
  whose gap grows 2e-5/frame gives `wrap_x = True` for a first-3-frame window and
  `False` over all 19. For built-in solids (closure static over the timeline) the
  flags are window-invariant; any animated/deforming grid hovering near the tolerance
  or actually opening/closing inside the video exposes it.
- **`surface_closed_axes` — CONFIRMED (b), conditional, same predicate** (`:371-400`):
  decides whether colour/material/normal maps are wrap-padded with an extra texel
  row/column (`:3101-3113`, `wrap_pad_texture` `:403-437`, colour path
  `:3114-3134`). Window-dependent padding changes how the sampler blends across the
  seam — a texture seam can appear/disappear at a batch boundary. Same probe:
  `closed_u True` (first 3 frames) vs `False` (full window).
- Monotone: reverse — a longer window can only fail the all-frames-under-tolerance
  predicate, so welding/wrap-padding can only be lost, never gained, by lengthening
  the window.

## Q6 — Remaining frame-axis reductions swept

Grep over `algan/rendering/` and `algan/mobs/` for
`.amax(0) .amax(dim=0) .max(dim=0) .max(0) .any(0) .any(dim=0) .all(0) .all(dim=0)
.amin(0) .min(dim=0) .min(0)` (kernels excluded — they take explicit frame counts):

| Hit | Classification |
|---|---|
| `primitives.py:3018-3021` bezier `amax(dim=(0,2))` + chunk max | **(b)** — Q1 |
| `scene_builder.py:566, 575, 576, 1063` promotion masks | **(b)** — Q3 |
| `primitives.py:352, 376, 383` shadow/closed-shell folds | (a)/(c) — inputs constant per mob today |
| `primitives.py:3071` `needs_endpoint = (~connection_visible).any(0)` | (c) — packed vertex count is an OR over frames (`:3056-3070`: discontinuous in ANY frame keeps the extra vertex; elsewhere it duplicates a linked vertex and contributes nothing); feeds merge widths → tie landing |
| `primitives.py:3379, 3381` circuit AABB inflation `amax(0)` | (c) — BVH bound tightness only |
| `primitives.py:2639` wedge solver `unresolved.any(0)` | (a) — work-list selection; per-frame sigma written per frame |
| `primitives.py:2210` diced width `counts.sum(1).amax()` | (c) — Q2 caveat 2c |
| `stbvh.py:662, 735` casts/opaque `all(0)` | (c) — documented-exact interval flags |
| `stbvh.py:792-793, 819-820`; `refit_bvh.py:301, 317-318, 436` | **Q4 structure** (pixel effect only via Q4's epsilon routes) |
| `scene_builder.py:963-967, 1158-1171, 1751, 1759, 1777-1779` | (c) — routing/layout, see Q3 |
| `scene_builder.py:2051` | (a) — vacuous (single-frame input) |
| `mobs/three_d_models/model_mob.py:410-413` | (a) — import-time vertex-axis bounds |
| `mobs/triangulated_bezier_circuit.py:190-191, 440, 590-591, 902` | (a) — construction-time point-axis reductions (tessellation/cache keys), not frame axes |
| `rendering/camera.py:393` | (a) — fit-to-mobs helper, point axis |
| `post_processing/anti_aliasing/smaa.py:497` | (a) — debug dump, batch-independent post-process |
| `surface.py:344-368, 371-400` welds/closed axes | **(b)** — Q5 |
| `surface.py:525-700` normals | (a) — Q5 |

Plus the two equality reductions that are batch predicates without value effect:
`_collapse_redundant_frames` (`primitives.py:2100-2129`) and `_dedup_time`
(`scene_builder.py:525-533`) — both (c).

### Consequence worth recording

Every (b) item re-decides when the **window** changes, and the render loop's
out-of-memory response is to shrink the frame window and retry
(`InsufficientMemoryException` at `scene_builder.py:503-504`; CLAUDE.md, Memory).
A retry therefore re-tessellates/re-promotes/re-welds partway through one video:
batch-size dependence is not only a cross-render reproducibility issue but a
potential intra-render pop at the retry boundary.

## Verification

Probes (`scratch_perf/ox/probe_batchwide_audit.py`, CPU only, no rendering):
Q1 monotone finer counts (128 vs 16/32/64/128); Q3 promotion flips between T=3 and
T≥6; `_dedup_time` collapses only equal frames; Q4 instance sets/tspans differ
structurally by window; Q5 `wrap_x`/`closed_u` True on the closed-phase sub-window,
False on the full window. All other claims are source-cited above; no GPU was used
and no tracked file was modified.
