# Read-only audit: launch/sync structure of the deterministic render thread

Scope: the default sheet route, `tracer.py:render_batch_raytraced` ->
`raytrace_render_wavefront` (sparse branch) -> `raster_pipeline.py:
prepare_sparse_raster_coverage` / `shade_sparse_raster_coverage` ->
`sheet_resolve_taichi.py:sheet_resolve_shade` + `_drain_sparse_secondary`
(the wavefront bounce loop). All claims are from source reading only; no
renders were run and no wall-clock numbers are quoted. Line references are
to the working tree as of this audit. Defaults assumed (verified in
`settings.py`): `SHEET_MASK_KERNEL`, `SHEET_RANK_KERNEL`,
`SHEET_ONE_MESH_KERNEL`, `SHEET_SAMPLE_DEPTH(_KERNEL)`, `SHEET_SHADE_SPLIT`,
`SHEET_POSITIONED_DEPTH`, `RASTER_OPAQUE_TRUNC_KERNEL`, `RASTER_PAIR_FLAGS`,
`RASTER_STRADDLE_CLIP`, `SHADOW_IDENTITY_REJECT`, `ANALYTIC_AA_TRI/BEZ/RUN`
all on; `POST_PROCESS_TONEMAP` on (`t_val == 3`,
`raytracing/settings.py:2740-2745`); `ANALYTIC_AA_SECONDARY_SAMPLES = 4`
(`settings.py:1848`); `MAX_SHADOW_LIGHTS = 16` (`shading_taichi.py:284`);
`KBUF = 4` (`raytrace_kernels_taichi.py:353`); 8 sub-pixel samples
(`_AA_NUM_SAMPLES`, `raster_taichi.py:230-234`).

---

## Claim 1 — host/device synchronisation points per render chunk

**CONFIRMED** (with one refinement: several "sync points" are implicit —
data-dependent output shapes — rather than literal `.item()` calls; listed
as such below).

### (i) Shadow-free opaque batch (`shadow_flag == 0`, nothing reflective ->
`refraction_flag == 0`, `pool_ratio == 1`)

Execution order within one `render_chunk` call:

1. **Chunk prologue** (`tracer.py:1555-1627`): arena bookkeeping,
   `_prefill_background` (torch fills only, no readback;
   `scene_builder.py`, `_prefill_background`). Env-map prefill kernel if
   applicable (`tracer.py:1595-1627`) — launch only.
2. **Wavefront entry** (`tracer.py:1708-1747`): batch metadata H2D copies
   via `_arena_values` / `_arena_copy` — async H2D, not syncs.
3. **`prepare_sparse_raster_coverage`** (`raster_pipeline.py:1402`):
   - Host torch pass: candidate pair derivation. With the precompute
     tables on, `_window_pairs` (`raster_pipeline.py:978-1034`) does
     index_select/clamp/where per chunk, then `_class_pairs_flat`
     (**SYNC**, twice per class pair present): `.nonzero()` at
     `raster_pipeline.py:948` and a data-dependent-length
     `torch.repeat_interleave` at `raster_pipeline.py:957` (the comment at
     `raster_pipeline.py:318-320` acknowledges the dynamic output length).
     `RASTER_PAIR_FLAGS`' per-batch host flags
     (`_class_any_flags`, `.cpu().tolist()` — itself a sync, but paid once
     per BATCH inside `_build_raster_tables`, `raster_pipeline.py:916-937`,
     `tracer.py:2521-2537`) skip classes proven empty.
   - KERNEL launches: `raster_bez_count` / `raster_tri_count`
     (`raster_pipeline.py:1629-1660`).
   - **SYNC** (explicit): `num_frags = int(counts64.sum().item())`,
     `raster_pipeline.py:1670`.
   - KERNEL launches: `raster_bez_write` / `raster_tri_write`
     (`raster_pipeline.py:1696-1736`).
   - Host torch passes: `_exact_fragment_order` (two argsorts +
     gathers, `raster_pipeline.py:1066-1096`), then
     `_gather_fragment_arrays` (fused-gather KERNEL when
     `RASTER_FUSED_GATHER`, else six `index_select`s;
     default OFF, `settings.py:1011`).
   - Opaque-prefix truncation (**SYNC ×3**): 
     `bool(opaque_s.any().item())` `raster_pipeline.py:1792`;
     `int(keep.sum().item())` `raster_pipeline.py:1794`; and
     `keep.nonzero(as_tuple=True)[0]` `raster_pipeline.py:1795` when
     truncated. (`_opaque_prefix_keep` is one KERNEL under the default-on
     `RASTER_OPAQUE_TRUNC_KERNEL`, `raster_pipeline.py:1148-1195`.)
   - **SYNC** (implicit, ×1-2):
     `torch.unique_consecutive(pix_s, return_counts=True)` —
     data-dependent output shape — `raster_pipeline.py:1786` and again at
     `raster_pipeline.py:1810` after truncation.
   - One-mesh caps: two KERNELs under default `SHEET_ONE_MESH_KERNEL`
     (`raster_pipeline.py:1222-1268`); no explicit readback.
   - Persistent copies into reverse-arena arrays
     (`raster_pipeline.py:1862-1879`).
   - **`compact_sheets`** (`sheets.py:789`) — host torch sort/reduce chain
     with these syncs in order:
     * KERNEL `sheet_conflict_rank` (default `SHEET_RANK_KERNEL`,
       `sheets.py:505-536`), then **SYNC** `int(rank.amax())`
       (`sheets.py:1078`; the truncation branch adds another reduction,
       `sheets.py:1080-1084`).
     * **SYNC** (implicit): `torch.unique(cid, sorted=True,
       return_inverse=True)` (`sheets.py:1088`), data-dependent output.
     * Default-on branches add: **SYNC** `int(shell_sid.amax().item())`
       and `int(seg[-1].item())` in the solid-shell ceiling
       (`sheets.py:1120`, `sheets.py:1148`; only with declared closed
       shells); KERNEL `mask_popcount` / `sheet_band_reduce`
       (`SHEET_MASK_KERNEL`, `sheets.py:362-445`) — no readback;
       **SYNC** `bool(is_tri.any())` inside `_shade_class`
       (`sheets.py:309`) and **SYNC** `bool(multi.any())` in
       `_sibling_weights` (`sheets.py:695`) under default
       `SHEET_SHADE_SPLIT`; **SYNC ×3** in the default-on
       `SHEET_SAMPLE_DEPTH` block: `int(band_of_sheet.max().item())`
       (`sheets.py:1388`), `enforcer.nonzero(...)` (`sheets.py:1421`),
       `bool(diff_sid.any())` (`sheets.py:1454`).
     * Diagnostic counters: **SYNC ×2**
       `int(((bands_per_group > 1) & tri_groups_mask).sum().item())` /
       `int(tri_groups_mask.sum().item())` (`sheets.py:1313-1314`),
       paid unconditionally.
4. `wf_finalize_uncovered` — skipped by default (`t_val == 3`;
   `tracer.py:2729-2753`).
5. **Tile loop** (`tracer.py:2830-3147`), per accepted tile attempt:
   - Arena allocs/fills (device-side `zero_`/`fill_`).
   - **`shade_sparse_raster_coverage`** (`raster_pipeline.py:2004`):
     * **SYNC** (explicit, once per chunk, memoized): 
       `so_host = coverage["sheet_offsets"].cpu()` at
       `raster_pipeline.py:2083-2086` — first tile attempt pays it; later
       attempts reuse `coverage["sheet_offsets_host"]`.
     * Single resolve launch, mode 0 (`raster_pipeline.py:2321-2336`).
   - Overflow check skipped (`pool_ratio == 1`, gate at
     `tracer.py:2982-2983`).
   - **SYNC** (per bounce iteration incl. the terminating one):
     `compactor.select` launches `compact_ray_slots`
     (`wavefront_kernels_taichi.py:104`) then reads the counter —
     `size = int(self.count.item())`, `tracer.py:823`. First call at
     `tracer.py:3002-3004` (seeds continuations), then every iteration of
     `_drain_sparse_secondary` (`tracer.py:2562-2696`: traverse KERNEL
     2567-2618, shade KERNEL 2619-2689, compaction 2690-2695).
   - **SYNC** (per accepted tile attempt): `_read_tile_alloc(rs_alloc)`
     = `rs_alloc.tolist()`, called at `tracer.py:3044`, defined
     `tracer.py:486-497`.
   - Composite KERNEL `wf_composite_accum_sparse`
     (`tracer.py:3133-3144`).
6. `ensure_render_headroom` — `mem_get_info` probe, `empty_cache()` only
   under pressure (`algan/utils/memory_utils.py:189-220`).
7. **Post-processing** (`tracer.py:1756-1763` →
   `post_process.py:297-383`): FXAA/bloom/tonemap kernels, then
   **SYNC** (whole-frame D2H): `frame_out.flip(-3).cpu()`
   (`post_process.py:399`) — the chunk boundary barrier.

### (ii) Shadowed reflective batch

Same list, plus:

1. The resolve runs TWICE per tile attempt: mode 1 event build
   (`raster_pipeline.py:2184-2199`) and mode 2 shade
   (`raster_pipeline.py:2298-2313`), separated by:
   - **SYNC**: `acc_idx = sheet_accept[:num_slice_sheets].
     nonzero(as_tuple=True)[0]` (`raster_pipeline.py:2200`); the following
     `index_select`/`scatter_` are device ops.
   - **SYNC** (default `SHADOW_IDENTITY_REJECT`):
     `_shadow_identity_epsilons(merged)` reads the WHOLE batch's
     `tri_pos` and ends in `(hi - lo).norm().item()` —
     `raster_pipeline.py:1379-1388`, called at `raster_pipeline.py:2235`.
   - KERNEL `raster_shadow_trace` between the two resolves
     (`raster_pipeline.py:2243-2297`, kernel at
     `raster_taichi.py:2744`).
2. A reflective batch under analytic coverage is a SPLITTING batch
   (`_secondary_split_needed`, `tracer.py:338-388`; ratio ≥ samples+1 via
   `_split_pool_ratio`, `tracer.py:391-445`), so the pool overflow check
   becomes live: **SYNC** per tile attempt BEFORE the drain —
   `int(rs_alloc[ALLOC_OVERFLOW].item())` (`tracer.py:2982-2984`).
3. On any overflow/OOM retry the whole attempt re-runs, so its syncs
   (`nonzero`, overflow `.item()`, `_read_tile_alloc`) recur
   (`tracer.py:2937-2976`, `3016-3059`).
4. If the glossy route is active (`glossy_reflection_mode() == 3`):
   **SYNC** once per chunk — `_gloss_frame_bounds(...).tolist()`
   (`tracer.py:2795-2799`, def `tracer.py:284-296`); per-frame-part
   scatter/composite/prefilter kernels otherwise launch-only.

---

## Claim 2 — bounce-loop sync cadence and absence of cross-chunk overlap

**CONFIRMED**, with two precisions.

- Per bounce iteration: exactly one forced readback — the compactor's
  `count.item()` (`tracer.py:823`) at the end of each
  `_drain_sparse_secondary` iteration (`tracer.py:2690-2695`) and of the
  seeding select (`tracer.py:3002-3004`). `active.numel()` itself is shape
  metadata, not a device read; the loop condition consumes the count the
  previous select already synced on. CONFIRMED.
- Per accepted tile: one `rs_alloc` readback — `_read_tile_alloc` =
  `.tolist()` at `tracer.py:3044`/`497`. Precision 1: a splitting batch
  additionally reads `rs_alloc[ALLOC_OVERFLOW].item()` once per ATTEMPT
  before the drain (`tracer.py:2983`), so rejected attempts pay a readback
  too (by design — the retry decision needs it). Precision 2: the docstring
  of `_read_tile_alloc` states this explicitly ("Looking costs one device
  synchronisation per tile, against the one the ray compactor already
  forces per wavefront iteration", `tracer.py:493-495`).
- No overlap across chunks inside the render thread: structurally, chunk N
  cannot begin until chunk N-1's `render_chunk` returns, and it returns
  only after `post_process_frames` copies the finished frames to the host
  (`post_process.py:383`, `.flip(-3).cpu()` at `post_process.py:399`) —
  a full-chunk device drain. Within a chunk, the per-iteration and
  per-tile readbacks above serialise host progress against device work
  anyway. The batch-prep prefetch is a separate single-worker executor
  doing torch-only preparation (`render_loop.py:2691-2754`, thread
  `algan-batch-prep` at 2753, "all torch-only" comment at 2707) — out of
  scope, as the brief says. CONFIRMED.

---

## Claim 3 — `_bezier_point_metrics` cost for a circuit candidate pixel

**REFUTED as stated.** A candidate pixel does NOT evaluate the metrics
against every segment of its circuit; the function prunes through two CSR
interval lists built by `bezier_acceleration.py`:

- Pruning that exists:
  1. **Scanline bins** (crossing parity): 16 y-bands per
     (edge-frame, circuit); an edge is registered in every bin its y-span
     touches (`bezier_acceleration.py:246-270`,
     `BEZIER_SCAN_BINS = 16` at `bezier_acceleration.py:22`). The query
     walks only the query point's own bin
     (`raytrace_kernels_taichi.py:527-543`).
  2. **Uniform 8×8 spatial grid** (nearest visible border edge): only
     border-visible edges whose endpoint AABBs overlap a cell are
     registered (`bezier_acceleration.py:272-313`,
     `BEZIER_SPATIAL_GRID = 8` at `bezier_acceleration.py:23`); the query
     visits only the cells its radius square touches, then applies the
     exact predicate — segment distance ≤ `query_radius` — per candidate
     (`raytrace_kernels_taichi.py:564-627`, exact test at 607).
  3. Circuit-level early-outs before either walk: v outside the circuit's
     [min_v, max_v] skips crossings entirely
     (`raytrace_kernels_taichi.py:527`); the query square missing
     [min_u, max_u]×[min_v, max_v] skips the border walk
     (`raytrace_kernels_taichi.py:564-567`).
  Both candidate sets are conservative and the exact predicates still run
  on candidates (the module and function docstrings say so:
  `bezier_acceleration.py:1-16`, `raytrace_kernels_taichi.py:489-509`),
  so worst case degenerates toward all-edges (a huge glyph spanning many
  cells/bins), but that is data-dependent, not structural.

- Call sites where it IS applied (all go through the pruning):
  - **Raster emission**: `_bez_pixel_hit`
    (`raster_taichi.py:1682`), reached from `raster_bez_count`
    (`raster_taichi.py:2080`) and `raster_bez_write`
    (`raster_taichi.py:2148`) — but only for pixels of candidate
    (circuit, bbox-chunk) pairs the host derived from screen bounds, i.e.
    bbox-gated before metrics run.
  - **Classic closest-hit traversal**: `_nearest_bezier_hit`
    (`raytrace_kernels_taichi.py:1292`) — per BVH leaf circuit whose plane
    is crossed, used by `_nearest_surface_g` (hence the ordered shadow
    march, `raytrace_kernels_taichi.py:2926`, and the dense route).
  - **Wavefront continuation hits (K-buffer gather)**: `_collect_hits`
    (`raytrace_kernels_taichi.py:2412`) — the event-batch traversal both
    bounce loops use, and shadow mode 4's gather
    (`raytrace_kernels_taichi.py:3112`).
  - **Shadow occlusion (any-hit)**: `_anyhit_opaque_bez`
    (`raytrace_kernels_taichi.py:2680`), reached from
    `_shadow_anyhit_opaque` (`raytrace_kernels_taichi.py:2701-2728+`) —
    shadow modes 3 and the deferred opaque check of mode 2
    (`raytrace_kernels_taichi.py:2804`, `3017`).

- Where it is NOT applied:
  - **The sheet resolve** (`sheet_resolve_shade`, both mode 1 and mode 2):
    neither pass re-evaluates edge metrics. They consume the emission's
    stored hit `(u, v)` and call only `_sample_circuit_color_blend` and
    `_bezier_normal` (`sheet_resolve_taichi.py:408-420`, `557-559`).
  - **`raster_shadow_trace`'s own body**: it traces occlusion via
    `_shadow_occluded` (`raster_taichi.py:3021`), so circuits are
    classified only where the dispatch reaches the any-hit/march/gather
    walkers above — there is no separate per-event metric evaluation.
  - Colour sampling (`_sample_circuit_color_blend`) and border-weight
    decoding never call it.

So the structural statement is: metrics ARE evaluated per candidate pixel
at emission and per candidate leaf-hit during traversal/shadow queries,
but always against a binned/grid-pruned candidate subset, never the full
segment list (except adversarial layouts).

---

## Claim 4 — double resolve launch on shadowed batches; item 9 memoization absent

**CONFIRMED.**

- `shade_sparse_raster_coverage` launches `sheet_resolve_shade` twice on
  any shadowed batch: mode 1 ("walks the IDENTICAL transport and writes
  one candidate shadow event per accepted lit triangle sheet") at
  `raster_pipeline.py:2184-2199`, and mode 2 ("the shading resolve reading
  the traced per-event visibility") at `raster_pipeline.py:2298-2313`,
  with `raster_shadow_trace` between them
  (`raster_pipeline.py:2243-2297`). Shadow-free batches take mode 0 once
  (`raster_pipeline.py:2321-2336`). This matches RENDERER_WORK_QUEUE.md
  item 9 ("The shadowed resolve runs the resolve kernel twice",
  `RENDERER_WORK_QUEUE.md:396-443`).
- Mode 2 re-fetches what mode 1 already fetched: the fetches
  (`_tri_color_g` line 427, `_tri_extra_g` line 430,
  `_tri_ior_transmission_g` line 465, `_tri_shadow_normals` line 500,
  `_pixel_footprint` line 532, plus the whole transport/corr/band/svis
  arithmetic lines 250-553) are NOT gated on `mode`; only `_shade_tri_hit`
  (gate `mode != 1`, line 434), the `shadow_vis` read-back (lines
  436-443), and the spawn paths (`mode != 1` gates throughout) differ.
  Item 9 documents exactly this and why the fetches cannot be cut from
  mode 1 (`RENDERER_WORK_QUEUE.md:408-426`).
- The item-9 memoization is NOT built: the event tables written by mode 1
  carry only position/normals/frame/mask/footprint/toffset
  (`event_pos/snrm/fnrm/frame/msk/dp/toff`, allocated
  `raster_pipeline.py:2175-2182`) — no colour/alpha/reflectivity/
  roughness/IOR/transmission payload that mode 2 could read back instead
  of re-fetching (~15 floats/sheet). Item 9's status is "new …
  Measurement needed" (`RENDERER_WORK_QUEUE.md:398-399`), i.e. scoped,
  unimplemented.
- One adjacent memoization DOES exist and should not be confused with it:
  the sheet-offsets CSR host copy is cached on the coverage dict
  (`coverage["sheet_offsets_host"]`, `raster_pipeline.py:2083-2088`) —
  that removes a per-tile `.cpu()`, not the mode-2 refetch.

---

## Claim 5 — the monolith kernels and their always-paid per-thread state

**CONFIRMED** (both are single `@ti.kernel` bodies carrying every material
path behind compile-time templates: `sheet_resolve_shade`'s signature
takes `frag_shading`, `refraction`, `ior_stack`, `sec_aa`, `glossy`,
`direct_spec`, `mode` etc. as `ti.template()`
(`sheet_resolve_taichi.py:110-184`); `wavefront_shade` likewise
(`wavefront_kernels_taichi.py:2363-2456`), including `frag_pipelines`/
`frag_scatters` composed pipelines and the per-material pid bitmask
`tri_pids`).

### Per-thread state the resolve walk carries (per covered pixel)

Locals live across the sheet loop (`sheet_resolve_taichi.py:193-269`):

| state | where | notes |
| --- | --- | --- |
| `acc` vec4, `weight` vec3 | 250-251 | transport |
| `svis` — `ti.Vector([1.0] * _AA_NUM_SAMPLES)` = 8 floats | 252 | per-sample transmittance |
| `mesh_ink`, `band_p`, `band_open`, `base_dist`, `bounces_left`, `processed`, `bounced`, `done`, `q` | 256-269 | scalars |
| `gl_px_per_rad`, `gl_taken`, `g_roff/g_aoff` | 220-248 | glossy-route only |
| dump locals | 217-219 | compile-time gated |

Inside the loop, per sheet: the material tuple `color/alpha/reflectivity/
rough/ior/T/albedo3` (389-395), `lvis = ti.Vector([1.0] * (3 *
MAX_SHADOW_LIGHTS))` — **48 floats** at the default 16 lights
(`sheet_resolve_taichi.py:396-401`, ceiling `shading_taichi.py:284`) —
plus the coverage bookkeeping `slots` (another 8-float vector, line 318),
`corr/cfac/dens/nsm`, band scalars, and the geometric normals
`normal/geo_normal` (554-555). Retirement writes ray state or atomic-adds
into `pix_accum` (1074-1121).

### Parts live only for shadowed/reflective/refractive sheets yet paid by every thread

Because it is one kernel body specialised by template, every variant
compiles the union of the branches its template allows, and register
pressure is set by the widest branch:

- **Shadowed-only**: the `lvis` vector is declared unconditionally at
  sheet scope "because the reflected lobe's direct-light add-back reads it
  again further down" (`sheet_resolve_taichi.py:396-401`) — a shadow-free
  batch (mode 0) still carries the declaration in every variant where
  `direct_spec != 0` can be compiled; the mode-2 `shadow_vis` read block
  (436-443) and the entire mode-1 event build (468-538) ride in every
  shadowed-batch variant of BOTH passes even though a given thread's sheet
  may be unlit/unaccepted.
- **Reflective-only**: `_material_reflectance`, the Fresnel/mirror-share
  arithmetic, `split_refl`, and all four reflection continuation blocks —
  glass split (675-796), glossy prefilter take (852-900), pane/split_refl
  (901-961), plain reflection (962-1015) — including the `sec_aa`
  jitter fan (`_jittered_surface_sample` loops, up to 32 taps
  unrolled at four sites; ceiling comment `raster_taichi.py:2185-2190`)
  and the `normal/geo_normal` fetches (554-576), which are needed only
  when `reflectivity >= 0 or T > 1e-4` (line 556) but are declared and
  branch-guarded, not absent, for matte sheets.
- **Refractive-only**: `is_glass/is_pane` classification (612-619),
  `_refract_ray`, `_offset_transmitted_origin`, transmitted spawns
  (675-739), and the nested-IOR stack plumbing handed to `_spawn_pool_ray`
  (wider `rs_sca` rows; `wavefront_kernels_taichi.py:94-98`).
- **Glossy-only**: the split-sum substitution and `pix_accum` extra-row
  writes (581-608, 852-900).

`wavefront_shade` mirrors the same shape: per-ray registers
`ro/rd/acc/weight/t_prev/layer_prev/seam_t/base_dist/bounces_left/
processed` plus the KBUF=4 six-vector k-buffer
(`wavefront_kernels_taichi.py:2511-2561`), the per-hit
`vis = ti.Vector([1.0] * (3 * MAX_SHADOW_LIGHTS))` (2674-2675),
reflection/refraction/IOR-stack scatter blocks (3058-3410, 3572+) and the
composed `frag_pipelines`/`frag_scatters` dispatch — all present in every
variant whose batch flags enable them, paid by threads whose rays never
take those branches.

---

## Question A — host torch passes remaining between kernel launches per chunk (sheet route)

In execution order, with the kernel launches they sit between (defaults as
stated in the header):

| # | host pass | site | forces sync? |
| --- | --- | --- | --- |
| 1 | candidate pair derivation (`_window_pairs` -> `_class_pairs_flat`: mask, nonzero, repeat_interleave, cumsum, stack) | `raster_pipeline.py:978-1034, 940-975` | YES — `.nonzero()` (948) + dynamic-shape `repeat_interleave` (957); up to 4× (bez/tri × opaque/trans), skipped per class when `RASTER_PAIR_FLAGS` proves empty |
| 2 | count cat/cumsum + fragment total | `raster_pipeline.py:1663-1670` | YES — `counts64.sum().item()` (1670) |
| 3 | fragment ordering + gather | `raster_pipeline.py:1742-1745` | no (argsort/index_select; fused gather is a kernel, default off) |
| 4 | opaque-prefix truncation decisions | `raster_pipeline.py:1792-1811` | YES — `any().item()` (1792), `sum().item()` (1794), `keep.nonzero()` (1795, when truncated) |
| 5 | covered-pixel discovery | `raster_pipeline.py:1786, 1810` | YES (implicit) — `unique_consecutive` output shape |
| 6 | conflict-rank ceiling check | `sheets.py:1077-1084` | YES — `int(rank.amax())` |
| 7 | band id compaction | `sheets.py:1088` | YES (implicit) — `torch.unique(return_inverse=True)` |
| 8 | solid-shell ceiling (declared closed shells only) | `sheets.py:1116-1186` | YES — `amax().item()` (1120), `seg[-1].item()` (1148), `nonzero(seg_start)` (1156) |
| 9 | shading-class sibling weights (default `SHEET_SHADE_SPLIT`) | `sheets.py:309, 695` | YES — two scalar `bool(tensor.any())` |
| 10 | sample-depth enforcer/subject pass (default on) | `sheets.py:1367-1510` | YES — `max().item()` (1388), `nonzero()` (1421), `bool(diff_sid.any())` (1454) |
| 11 | diagnostic group counters | `sheets.py:1313-1314` | YES ×2 — `.item()` sums |
| 12 | sheet-offsets host copy (first tile attempt of the chunk only) | `raster_pipeline.py:2083-2086` | YES — `.cpu()`, memoized per coverage dict |
| 13 | shadow event compaction (shadowed batches) | `raster_pipeline.py:2200-2240` | YES — `nonzero()` (2200); identity epsilons `.item()` (1386 via 2235) |
| 14 | pool overflow probe (splitting batches) | `tracer.py:2982-2984` | YES — `.item()` |
| 15 | ray compaction count (every bounce iteration) | `tracer.py:3002-3004, 2690-2695, 823` | YES — `count.item()` |
| 16 | tile allocator readback (every accepted attempt) | `tracer.py:3044, 486-497` | YES — `.tolist()` |
| 17 | frame-buffer download (chunk boundary) | `post_process.py:383, 399` | YES — `.cpu()` |

Everything else between launches is pure device torch (fills, gathers,
cumsum, searchsorted) or arena pointer arithmetic
(`memory_utils.py:660-674` is host-only).

## Question B — per-chunk work consuming provably batch-invariant inputs

- **Already fixed (the big one)**: projection/bounds tables
  (`tri_screen`, `tri_bounds`, `bez_bounds`) cover the whole prepared
  batch and are built once per batch from the arena's persistent end,
  cached on `merged["_raster_tables"]` (`tracer.py:2506-2537`,
  `_build_raster_tables` at `tracer.py:1029-1100`). Later chunks reuse
  them; only the row-band clamp/chunk expansion (`_window_pairs`) is
  redone per chunk, which IS chunk-dependent. Camera and light tables are
  likewise allocated once per batch (`tracer.py:1239-1266`, `1474-1480`).
- **Batch-invariant values rebuilt per chunk (small)**:
  `gen_meta` `[0.5, 0.5, half_w, half_h]` and `layer_offsets_t`
  (layer offsets/env placement/far clip/max bounces) are re-materialised
  into the arena on every `raytrace_render_wavefront` call although their
  values depend only on batch-wide facts
  (`tracer.py:2453-2501`); the `col_row_arr` placeholder is re-zeroed per
  call (`tracer.py:2346-2348`). Cost is a handful of floats — negligible
  bandwidth, though each is an extra tiny H2D per chunk.
- **Batch-invariant value recomputed per TILE attempt (not just per
  chunk)**: `_shadow_identity_epsilons` reduces the ENTIRE batch's
  `tri_pos` to a scene diagonal and synchronises on `.item()`
  (`raster_pipeline.py:1379-1388`) at every resolve call
  (`raster_pipeline.py:2235`) — the answer cannot change within a batch
  (`merged["tri_pos"]` is fixed), yet a multi-tile shadowed chunk pays the
  full-array amax/amin reduction and a sync per attempt. This is also a
  hidden entry in claim 1(ii)'s sync list.
- **Per-chunk recomputation that is genuinely chunk-dependent** (listed to
  delimit the finding): the COUNT/WRITE emission, fragment sort/truncation,
  sheet compaction, and the covered_idx/run_offsets/sheet CSR all derive
  from the chunk's frame window and must rerun; their inputs
  (`merged` geometry/BVH/edge tables, `tri_screen`) being batch-wide does
  not make the outputs reusable.
- Fallback-only: with `RASTER_TRI_PRECOMPUTE`/`RASTER_BEZ_PRECOMPUTE`
  off, `_frame_pairs`/`_frame_bez_pairs` re-derive screen bboxes per
  frame per chunk from the (batch-wide) projection table
  (`raster_pipeline.py:1455-1500`) — the per-batch batching the precompute
  exists for would be lost, but defaults keep it on.

---

## What I did not verify

- Any measurement: no renders, no profiling, no wall-clock or device-time
  figures were taken; the container discipline (CPU-only, no timing
  claims) was observed. Whether the syncs above dominate runtime is
  unstated — item 9's own caveat ("Measurement needed") applies to every
  ranking question here.
- CUDA stream semantics: I asserted ordering/synchronisation effects of
  mixed Taichi/torch launches from the code's readback structure, not
  from a stream-level analysis of how Taichi's CUDA backend serialises
  against torch's streams. If they share no stream, some "implicit"
  barriers could behave differently than assumed.
- The Monte Carlo (`samples > 1`), textured, sorted-material, dense-tile
  (`run_tile`/`_run_wavefront_tiles`) and `_drain_*` variants other than
  the sparse drain were skimmed only where they share code paths cited
  here; their sync profiles were not audited.
- `ManualMemory.get_tensor` internals (whether an allocation can trigger
  `empty_cache`/sync under pressure) were not traced; scope/temp/set_pointers
  were treated as host bookkeeping per their implementations at
  `memory_utils.py:655-674, 800-803`.
- Behaviour of `torch.repeat_interleave`/`unique*`/`nonzero` as implicit
  sync points is asserted from PyTorch's documented data-dependent-output
  semantics, not verified against a specific build.
- The exact register/liveness cost inside compiled kernels (claim 5) is a
  source-level reading; actual PTX register allocation, spills, and the
  cost the unchosen template arms impose were not inspected in compiler
  output.
- Interaction with the daemon, prefetch worker timing, and whether any
  OTHER concurrent audit session modified files while I read (tree shared
  with other sessions per the brief) — line numbers reflect one snapshot.
