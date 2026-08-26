# Read-only audit: time expansion and geometry duplication in the scene merge

Audit of `scratch_perf/ox/brief_struct_merge.md`. Read-only: no file outside this
report was modified, no renders, no pytest, no timing claims. Every claim below
was checked against source; citations are `file:line` in the repo tree at HEAD.
Line numbers for the two biggest files were verified against the working copy.

Framing from the plan of record: DESIGN_optimization_targets.md T4 records that
"`scene_builder`'s `_cat_collections` runs `_unify_time` over the primitives it
concatenates, so a single moving flat mesh anywhere in the scene expands the
static one straight back out at merge time" (DESIGN_optimization_targets.md:752-761),
and that T4's frame collapse is a within-dice-call optimization
(DESIGN_optimization_targets.md:651-700). `MERGE_DEDUP_TIME` is default on
(algan/rendering/raytracing/settings.py:488-490).

---

## Claim 1 — `_unify_time` expands static collections to the batch's frame count

**CONFIRMED**, with two precision notes.

Mechanism:

- `_expand_frames(x, num_frames)` returns `x.expand(...)` when `x.shape[0] != num_frames`
  — a stride-0 view, no copy (algan/rendering/raytracing/utils.py:11-14).
- `_unify_time(tensors)` takes `T = max(t.shape[0])` over the group and expands
  every member whose leading dim is 1 to T (utils.py:37-48).
- `_cat_collections(tensors, dim)` calls `_unify_time` **only when there are ≥2
  collections** (`len(tensors) == 1` passes through uncopied, utils.py:57-58),
  then materializes everything with `torch.cat(tensors, dim).contiguous()`
  (utils.py:59-60). The stride-0 views are free until this cat: the physical
  cost is T rows written per collection, including the static one.

So: expansion fires whenever ≥2 collections of the same geometry type are merged
and any one of them carries more than one frame. Note that a *lone* collection is
not saved either — timeline materialization hands attributes back as a dense
`[T, rows, D]` window with one row per frame whether or not the mob moved
(`active_state = _query_row_states(times, ...)`, algan/animation_timeline/timeline.py:1529-1531;
the non-compact path likewise, timeline.py:1543-1549; "Materialization hands a
mob's attributes back one row per frame even when the mob does not move",
algan/rendering/raytracing/primitives.py:2104-2106) — and
`_pack_projected_flat_geometry` makes its packed arrays dense with `.contiguous()`
(primitives.py:1134-1139). The T identical rows exist before the merge ever runs;
the merge's contribution is preventing them from ever going back to 1.

Merged fields that go through the expanding concat (all calls in
`_merge_scene`, algan/rendering/raytracing/scene_builder.py):

| field | call site | shape | bytes/(frame·unit), f32=4B |
| --- | --- | --- | --- |
| `tri_pos` | scene_builder.py:1493 | `[T, Ntri, 9]` | 36 |
| `tri_norm` | :1494 | `[T', Ntri, 9]` | 36 |
| `tri_mat_id` | :1495-1497 | `[1, Ntri]` i32 (pack-time shape, primitives.py:833-835) | 4 (no time axis at pack) |
| `tri_mat` | :1502 via `_cat_mat_blocks` | `[T', Ntri, 34]` built-in (`MAT_W = 34`, algan/rendering/raytracing/shading_taichi.py:78; block allocated at that width, primitives.py:875-880); custom pipelines pad to the widest W (utils.py:63-84) | 136 |
| `tri_frame_lo` / `hi` | :1503-1504 | `[T, Ntri, 3]` each | 24 + 24 |
| `tri_frame_opaque` / `casts` | :1505-1506 | `[T, Ntri]` / `[1, Ntri]` bool | 1 (+casts has no time axis, primitives.py:317-353) |
| `tri_colors` | :1564-1566 | `[T', Ntri, 3, 5]` | 60 |
| `tri_extra` | :1566 | `[T', Ntri, 15]` (`_EXTRA_W = 15`, algan/rendering/raytracing/raytrace_kernels_taichi.py:375) | 60 |
| `tri_uvs` | :1623-1625 | `[T'', Nuv, 6]` (textured/promoted tier only) | 24 |
| `tri_obj` | :1478-1482 | `[1 or T, Ntri]` i32 — per-frame only when a diced-PN collection is in the batch (primitives.py:1157-1179, 2250-2273) | 4 when expanded |
| `tri_closed` | :1490-1492 | `[1, Ntri]` f32 (every part is single-frame by construction, primitives.py:356-389) | never expands |
| `circuit_meta` | :1810-1812 | `[T', Ncirc, 24]` (`_M_WIDTH = 24`, raytrace_kernels_taichi.py:425) | 96/(f·circuit) |
| `circuit_colors`, `circuit_border_colors` | :1813-1818 via `_cat_circuit_color_grids` (pads grid width to the max first, scene_builder.py:51-71) | `[T', Ncirc, P, 5]` | 20·P each |
| `edges_2d` | :1819-1821 | `[T', Nedge, 6]` | 24/(f·edge) |
| bez `frame_lo`/`hi`/`opaque`/`casts` | :1852-1859 | `[T, Ncirc, 3]` ×2 + masks | 48 + masks |
| `textures` | :1721-1724 | `[max T' of maps, total texels, 5]` | 20/texel |

Additional same-shape expansion sites *inside* one primitive (same
`_unify_time` all-or-nothing, so one animated member of the group drags the rest):

- triangle surface params: reflectivity/roughness/IOR/transmission/sigma unified
  together (primitives.py:974-992);
- bezier geometry: polyline verts, plane centers/bases, degeneracy and border
  flags unified together (primitives.py:3116-3138); circuit metadata likewise
  (primitives.py:3207-3241); bounds/colors unify (primitives.py:3319-3343);
- per-frame light rows: every light expanded to the batch's `num_frames` then
  `stack(...).contiguous()` (scene_builder.py:2117-2128, 2140-2154).

**Which fields stay expanded under `MERGE_DEDUP_TIME`** — see claim 2; short
version: `tri_pos` deliberately never collapses (scene_builder.py:1637-1639), the
bounds/opaque/casts/valid tables and `tri_obj` are not in the dedup lists at all,
and the six collapsible triangle tables + four bezier tables survive at full T
whenever *any* collection animates that particular property.

## Claim 2 — what `MERGE_DEDUP_TIME` actually does

**CONFIRMED** (as a partial remedy; it does not undo claim 1 in general).

- Mechanism: after concatenation, `_dedup_time(x)` tests the **whole tensor**
  against its own first frame — `x.shape[0] > 1 and bool((x == x[:1]).all())` —
  and if equal keeps `x[:1].contiguous()` (scene_builder.py:525-533). Consumers
  read the time axis as `f % shape[0]` (kernels) or `_expand_frames` /
  `index_select(0, f % len)` (host tables), so a length-1 axis serves every frame
  (comment at scene_builder.py:1632-1638).
- Granularity: one equality test per table, post-cat, leading axis only.
- Key: none. No hashing, no grouping, no per-row or per-collection structure —
  a single boolean over all frames × all prims × all channels of the concatenated
  array.
- Applied to exactly ten tables (scene_builder.py:1639-1648 triangles:
  `tri_norm`, `tri_mat_id`, `tri_mat`, `tri_colors`, `tri_extra`, `tri_uvs`;
  :1829-1836 beziers: `circuit_meta`, `circuit_colors`,
  `circuit_border_colors`, `edges_2d`). Toggle:
  settings.py:488-490 (`ALGAN_MERGE_DEDUP_TIME`, default True), setter
  `set_merge_dedup_time` settings.py:591-596.

What it does **not** collapse:

- Any table in which *any* collection animated that property: because the test
  runs on the concatenated array, one colour-tweening mob anywhere in the batch
  keeps `tri_colors` at `[T, N_all, 3, 5]` — the static mobs' rows inside it stay
  T byte-identical copies. This is precisely "a static mesh merged beside a
  moving one" for every per-vertex property.
- `tri_pos` — deliberately excluded ("rigid motion lives in tri_pos, which is
  deliberately not collapsed", scene_builder.py:1637-1639). A scene whose motion
  is purely rigid still stores every static mesh's positions T times beside the
  mover's.
- The per-frame bound/mask tables (`tri_frame_lo/hi/opaque/casts/valid`,
  `bez_*` equivalents; built at :1503-1506/:1654-1656 and :1852-1859/:1865-1869)
  — never candidates. Even a fully static mesh's AABB rows are stored T times
  once anything else in the batch has T rows (or simply because pack-time emitted
  T rows; see claim 1's precision note).
- `tri_obj` (:1478-1482): stays `[T, N_all]` whenever a diced-PN collection is in
  the batch, even though every flat collection's ids are frame-constant `[1, N]`.
- The assembled `textures` buffer (:1721-1724) — individual maps are deduped
  only where promotion produced them via `_dedup_time` (:606-618); a map appended
  raw from `_stash_texture_maps` (primitives.py:1094-1098) carries whatever frame
  count materialization gave it, and `_cat_collections` widens all maps to the max.

## Claim 3 — projection and screen-bounds tables run over full [T, N]

**CONFIRMED. No code detects the static-camera-and-static-row case.**

- `precompute_triangle_projection` sizes the table
  `frames = max(tri_pos.shape[0], cam_origin.shape[0], screen_point.shape[0],
  pixel_basis_x/y.shape[0], tri_frame_valid.shape[0])`
  (algan/rendering/raytracing/raster_pipeline.py:220-227) and projects **every**
  (frame, triangle): gather each source at `frame_ids % src.shape[0]`
  (:241-248), then the full plane/projection arithmetic over
  `[frames, ntri, 3]` (:249-303). Its own comment states the design intent:
  inputs may be deduplicated to T=1 independently and the table must span the
  longest dynamic input (:216-219) — i.e., a static mesh whose `tri_pos` was
  *not* collapsed (always the case, claim 2) gets projected once per frame even
  under a frozen camera, producing T identical rows.
- `precompute_triangle_screen_bounds` — same pattern over
  `frames = max(tri_screen/valid/opaque/uncertain lengths)`
  (raster_pipeline.py:843-848, gathers :851, :866-876, :889-894).
- `precompute_circuit_screen_bounds` — same over eight sources
  (raster_pipeline.py:720-738, :786-787), projecting all eight AABB corners of
  every circuit for every frame (:740-760).
- All three are called once per batch from `_build_raster_tables`
  (algan/rendering/raytracing/tracer.py:1029-1100, call sites :1057, :1074,
  :1088) into persistent arena tensors.

What *does* exist nearby, and why it is not this detection:

- The modulo indexing itself (`f % shape[0]`) avoids re-*reading* distinct
  memory for a collapsed input but never skips recompute, and nothing feeds it a
  collapsed `tri_pos`.
- `_class_any_flags` skips whole frames/classes with no candidates
  (raster_pipeline.py:916-937, consumed at :992-1005) — an emptiness skip keyed
  on visibility, not a constancy skip.
- Stride-0 static detection exists elsewhere (`_frame_broadcast_base`,
  primitives.py:154-166) but only for the PN/bezier level-search kernels, never
  here.

## Claim 4 — PN surfaces and bezier circuits rebuild per batch

**CONFIRMED for both. Nothing caches diced microtriangles or sampled circuits
across batches; T4's collapse is within a single dice call.**

Per-batch lifecycle: each batch fetches fresh primitive objects —
`get_batch_of_primitives` walks actors and calls `actor.get_render_primitives()`
per batch (algan/render_loop.py:2124-2153, construction e.g.
algan/mobs/pn_mesh.py:80-102 building a new `LogicalPNTrianglePrimitive` from the
window's materialized state); `project_to_screen` runs per batch either on the
render thread (render_loop.py:1272-1293) or the prep worker (render_loop.py:2437).
There is no persistence of projected output between batches: the merged scene is
cached on `primitives[0]._rt_merged_scene` for the life of the batch only
(scene_builder.py:1250-1253, released with the batch).

Logical PN:

- `_dice_logical_pn` runs the level search and the write-out from scratch every
  call: control nets rebuilt (:2162-2180), levels searched (:2191-2202), and the
  diced outputs freshly zero-allocated as dense `[num_frames, max_triangles, 3, C]`
  (`allocate`, primitives.py:2290-2309) then scattered per (frame, patch)
  (:2421-2444). `self.corners/normals/colors/...` are replaced by these dense
  arrays (:2446-2452) and repacked dense by `_pack_projected_flat_geometry`
  (:1134-1142).
- The shipped T4 optimization collapses the *source* rows within this one call
  (`_collapse_redundant_frames`, primitives.py:2099-2129; gate at :2141-2149)
  and dedups patch evaluation across frames within the work list
  (`_PatchChunk`, primitives.py:83-133; used at :2384-2420) — both scoped to the
  current `(levels, frames)` selection. DESIGN says the same: "a mesh that holds
  still ... reaches it as T byte-identical copies ... and was diced T times over",
  fixed per call (DESIGN_optimization_targets.md:653-682). Across batches there
  is no memo: the next batch re-searches levels and re-evaluates every patch even
  for a mesh whose corners, normals, camera and resolution are unchanged.
- What *is* cached across batches is topology only: dice patterns, vertex UVs,
  triangle indices, boundary structures (`_DICE_PATTERN_CACHE` etc.,
  algan/rendering/logical_pn.py:375-668) and constant sample-weight tensors
  (`_SAMPLE_TENSOR_CACHE`, primitives.py:65-80). No geometry values.

Bezier circuits:

- `project_to_screen` re-runs the chord-count search per batch
  (`_compute_samples_per_segment`, primitives.py:2873-3031) — note the search
  reduces the error **over all frames of the batch**
  (`error_squared.amax(dim=(0, 2))` → one chord count per segment for the worst
  frame, :3018, :3026-3029), so the result is batch-window-dependent by
  construction and could not be reused verbatim across windows without changing
  output. Then `_build_circuit_geometry` resamples and repacks everything
  (primitives.py:3033-3279) and `_build_frame_bounds` recomputes AABBs
  (:3281-3383). No cross-batch caching of chord counts, polylines, or edge
  acceleration exists; `build_bezier_edge_acceleration` also rebuilds its
  per-(frame, circuit) scanline tables from scratch each batch
  (algan/rendering/raytracing/bezier_acceleration.py:117-153, :184-230, :327-328).

## Claim 5 — consumers that assume a dense time axis

**CONFIRMED premise.** The merge produces dense `[T, N]` arrays; a per-collection
stride-0 layout (static collections stored at length 1 inside one concatenated
array) would need an indirection (per-primitive time base/stride, or per-array
row remaps) at every site below. Enumerated:

Taichi kernels — every geometry read resolves time as one global modulo
`f % arr.shape[0]`, which supports a length-1 axis but not mixed strides within
an array:

- `raytrace_kernels_taichi.py`: `tri_pos` reads :1066, :1739, :1792, :2155,
  :2484; `tri_obj` :1033; `tri_colors` :1496, :1510, :1578; `tri_uvs` :1588;
  `tri_extra` :1604, :1667, :1717; `tri_norm` :1732; `textures` :1537;
  `circuit_meta` :1325, :1447, :1678, :1752; `edges_2d` :1289; circuit color
  grids :1326, :1448; refit-BVH frame-row base
  `(f % (blocks.shape[0] // num_blocks)) * num_blocks` :821-828. Shadow paths
  consume the same arrays through `_anyhit_opaque_tri`/`bez`
  (:2461/:2571, reading `tri_pos`/`circuit_meta` at :2484, :2644, :2730 and
  meta/edges at :2644-2677) and the ordered march/gather kernels; the path
  tracers likewise (:3374, :3695).
- `raster_taichi.py`: `tri_pos` :897, :2526, :2644, :2730; `tri_screen` rows
  :913-921 (table indexed at absolute frame `ts`); `circuit_meta` :1643;
  `edges_2d` :1680; `pixel_world_scale` :3025.
- `wavefront_kernels_taichi.py`: :463, :594, :2268, :2648, :2749, :3050, :3078,
  :3451, :3474 (`tri_pos`, `circuit_meta` modulos).
- `sheet_resolve_taichi.py`: :412 (`circuit_meta`), :566, :680 (`tri_pos`).

Host-side torch consumers:

- `precompute_triangle_projection` / `precompute_triangle_screen_bounds` /
  `precompute_circuit_screen_bounds` (raster_pipeline.py:185-304, 809-913,
  676-806) — build their tables over the full expanded span with per-source
  modulo gathers (claim 3).
- Sheet resolve host tables: `_rows(arr, frame_rel, time_start)` applies the
  global `(f_rel + time_start) % arr.shape[0]` convention
  (algan/rendering/raytracing/sheets.py:288-292) for `tri_norm` :316, `tri_pos`
  :338, :567, cameras :568, `tri_screen` :584, `pixel_world_scale` :596.
- STBVH build: consumes `[Tc, N, 3]` bounds and segments instances over the
  whole axis (`segment_primitives_in_time`, stbvh.py:741-743; opaque prefix sums
  over Tc :750-752; Morton time normalization over `num_frames` :795, :822-826).
  It has a `Tc == 1` branch that spans one instance across
  `[0, num_frames - 1]` (stbvh.py:692-695, :728-735) — but real batches can
  never take it, because pack-time emits T rows (see Q-B item 4) and the builder
  raises unless `Tc == num_frames` (stbvh.py:736-740).
- Refit BVH build: same input contract (`Tc ∈ {1, num_frames}`,
  refit_bvh.py:291-298; docstring advertises the Tc=1 dedupe :284-289) and
  refits child boxes for **all Tb frames** into `[Tb * B, 8, ARITY]` blocks
  (refit_bvh.py:420-514; per-frame leaf opacity :472-474). Kernel side mirrors
  it with per-frame row bases (raytrace_kernels_taichi.py:821-828).
- Bezier edge acceleration: builds `num_frames × num_circuits` record groups and
  repeats edge offsets per frame (bezier_acceleration.py:147, :184-230, :327-328);
  header stride is `f % edges_2d.shape[0]` by construction
  (scene_builder.py:1822-1828).
- Merge-time derivatives computed over the expanded arrays: visibility masks
  `(hi >= lo).all(-1)` (scene_builder.py:1654, 1865), memory-trim re-layout
  (index-selects prims, leaves the time axis intact, :987-1014), textured-bank
  promotion `_build_textured_scene` (`_expand_frames` to T first, :1139-1171).
- Camera/light packing: `_flat_frames` + `_expand_frames` to `num_frames`
  (tracer.py:1234-1246; scene_builder.py:2117-2154).

## Question A — largest allocations and pure broadcasts

For a mixed static+moving batch of N triangles / C circuits over T frames
(bytes per (frame, unit) from the claim-1 table):

Largest merged per-batch allocations, ranked:

1. `tri_mat` — 136 B/(f·tri). Survives at full T whenever *any* material slot
   animates anywhere (including `one_sided`/`no_shadow_receive`, which ride the
   block, primitives.py:864-873).
2. `tri_colors` and `tri_extra` — 60 B/(f·tri) each; survive at full T if any
   colour/surface-param animates anywhere.
3. `tri_pos` — 36 B/(f·tri), always full-T when anything moves (never deduped).
4. `tri_norm` — 36 B/(f·tri), survives if anything deforms or rotates enough to
   change shading normals.
5. Bounds+flags — ~50 B/(f·tri) total (`lo`+`hi` 48, opaque/casts/valid ~3),
   never deduped.
6. Per-circuit: `circuit_meta` 96 B/(f·circ); the two color grids 20·P B each
   (P = padded grid width — text glyphs can make P large); `edges_2d`
   24 B/(f·edge); bounds ~50 B/(f·circ). Plus `edge_accel`, which fans
   records out per frame (bezier_acceleration.py:217-230).
7. Outside the six-table collapse: the `textures` buffer (20 B/texel × frames of
   the widest map appended) and, before the merge, the diced-PN attribute arrays
   (~10 dense `[T, M, 3, C]` tensors, primitives.py:2302-2320).

Pure time-broadcasts of one row (rows that are byte-identical copies of the
collection's frame-0 row, kept only because they sit in an expanded merged
array):

- In a rigid-motion-only mixed scene, the survivors are exactly the never-deduped
  tables: every static mesh's `tri_pos` rows and its bound/flag rows, plus
  `tri_obj` rows of flat collections when a diced-PN collection forces `[T, N]`.
- Once any mesh animates colour/material/normals/uvs, that table stops
  collapsing batch-wide, and *every* static mesh's rows in that specific table
  become pure broadcasts (e.g. one tweening mob turns all N_static × T
  `tri_colors` rows — 60 B each — into copies of one row).
- Static lights under `_pack_lights`: position/color rows broadcast to
  `[T, L, ·]` and materialized (scene_builder.py:2117-2128, 2140-2154).

## Question B — static-ness signals computed, then discarded

Yes — five signals, each dying at a specific place. (`distinct_frames` as a name
exists only in DESIGN prose describing a measurement
(DESIGN_optimization_targets.md:659-660); there is no such symbol in `algan/`.)

1. **`geometry_static` / `_collapse_redundant_frames`** (primitives.py:2099-2149):
   detects a frame-invariant PN source, keeps one control net, enables patch-major
   ordering and per-patch evaluation dedup (:2357-2384). Dies at `allocate()` +
   scatter: the outputs are dense `[num_frames, max_triangles, …]` regardless
   (:2290-2309, :2421-2444), re-flattened dense at :1134-1139, and (for multi-
   collection batches) re-expanded by `_cat_collections` anyway.
2. **`_frame_broadcast_base`** stride-0 detection (primitives.py:154-166), feeding
   the PN criterion kernels (:198-232) and bezier chord kernel (:181-195): avoids
   uploading T copies of a static net into the level searches. Dies at the
   searches' outputs: levels/chord counts are per-frame decisions and the
   write-out is dense; the signal is not propagated to the merge.
3. **`_dedup_time`** (scene_builder.py:525-533): computes exact whole-table
   constancy — then discards the *reason* it held. It cannot express "these
   columns are constant, those are not", so one animated collection voids the
   collapse for every static row in the table (claims 1-2).
4. **The BVH builders' structural static path**: `build_stbvh` documents and
   implements `Tc == 1 → one instance spanning all frames` (stbvh.py:692-695,
   :728-735) and `build_refit_bvh` accepts `Tc == 1` ("the refit dedupes to one
   time slice", refit_bvh.py:284-298). Both are dead code for real batches:
   `_pack_frame_visibility` unifies bounds against the corners' frame count, so
   `frame_lo/hi` always carry T rows (primitives.py:1060-1076; bezier analogue
   :3319-3366), and the builders raise on any other mismatch (stbvh.py:736-740,
   refit_bvh.py:295-298). Static geometry arrives at the trees pre-expanded.
5. **Frame-valid masks** (`tri_frame_valid`/`bez_frame_valid`,
   scene_builder.py:1654, :1865; consumed at raster_pipeline.py:583-586,
   :634-635, :786-787 and by BVH validity stbvh.py:729): they already answer
   "is this (frame, prim) cell live?" per frame — the natural carrier for a
   sparse/static encoding — but they key on visibility (alpha), not value
   constancy, and are discarded with the batch.

The enabling fact on the consumer side: every kernel already reads time modulo
each array's own length (`f % shape[0]`), so length-1 axes are universally legal
today — what no consumer supports is a *mixed-stride* merged array (claim 5).

---

## What I did not verify

- **No measurements.** CPU-only, read-only audit per the brief: I made no
  wall-clock, allocation-size, or profile claims about real scenes; the byte
  figures above are arithmetic from declared widths/dtypes, not measured.
- **No execution of any kind** — no renders, no pytest, no probes; behaviour is
  argued from source only.
- **Exhaustiveness of kernel read sites**: the Taichi sites in claim 5 were
  collected by pattern (`f % <array>.shape[0]` and friends) over the four kernel
  modules plus sheets; a consumer that indexes time by some other spelling
  (e.g. a precomputed absolute frame index like `tri_screen[ts, …]` at
  raster_taichi.py:913-921) may exist beyond those listed.
- **Stride-0 provenance**: I did not trace whether today's timeline can hand a
  *stride-0* (expanded-view) attribute window to a mob in any current code path;
  `_collapse_redundant_frames`' `stride(0) == 0` branch (primitives.py:2120-2121)
  handles it defensively, but my report assumes dense windows per the
  materialization code and DESIGN.
- **Animated texture maps**: the textures-buffer expansion claim assumes an
  animated/video texture materializes as `[T, H·W, C]`; I read `_append_texture`
  and `_stash_texture_maps` but did not walk ImageMob/video-texture
  materialization end-to-end.
- **CUDA-specific behaviour**: merge-on-gpu upload, arena copying, and NVENC
  paths were not examined for shape-dependent differences; all analysis is
  device-agnostic torch semantics.
- **History**: I did not check git history for when `MERGE_DEDUP_TIME`, the
  `Tc == 1` BVH branches, or the hot/cold split landed, nor whether the three
  tests DESIGN says assert a per-frame diced contract still do.
- **Other reports in this directory**: other audit sessions share this tree; I
  did not read or reconcile `REPORT_*.md` files beyond confirming filenames.
