# Read-only audit: acceleration structures on the default deterministic path

Scope: the brief in `scratch_perf/ox/brief_struct_accel.md`, against
`algan/rendering/raytracing/{stbvh.py, refit_bvh.py, scene_builder.py, tracer.py,
raster_taichi.py, wavefront_kernels_taichi.py, raytrace_kernels_taichi.py,
settings.py}` and `DESIGN_hybrid_raster.md` §9/§13 (all paths under
`algan/rendering/raytracing/` unless stated). No files were modified; no renders
or pytest runs; no wall-clock claims (CPU-only container).

Defaults snapshot read from source (the design doc's §11 default table is stale;
source wins per CLAUDE.md): `BVH_REFIT` ON (`settings.py:448`),
`BVH_DEFER` ON (`settings.py:469`), `HYBRID_RASTER` ON (`settings.py:870`),
sheet-route toggles ON (`SHEET_RESOLVE settings.py:1103`, `ANALYTIC_AA:1369`,
`ANALYTIC_AA_RUN:1488`, `RASTER_SPARSE_COVERAGE:1079`,
`RASTER_EMPTY_SKIP`/`RASTER_COVERED_SHADE` per `tracer.py:573-575`),
`SAMPLES_PER_PIXEL = 1` (`settings.py:33`), `SHADOWS` OFF (`settings.py:2549`),
`SHADOW_ANYHIT` False (`settings.py:620-624`), `WF_OPAQUE_CLOSEST`/`_PREPASS`
OFF (`settings.py:126-127`), `OPAQUE_BVH_SKIP_DEAD` ON (`settings.py:837`),
`BLOCK_F16` ON (`stbvh.py:96`), `BVH_ARITY` 4 (`stbvh.py:75`),
`GATE_EMPTY_TRAVERSALS` True (`settings.py:119`), `SOFT_SHADOW_SAMPLES` 8
(`algan/settings/_startup.py:81`). "Default-path batch" below means: SPP==1,
sheet route engaged, wavefront continuations for bounces — i.e. what
`analytic_raster_route_active` (`tracer.py:542-635`) plus `use_raster`
(`tracer.py:2389-2402`) select.

---

## Claim 1 — every tree a default-path batch builds is a RefitBVH

**CONFIRMED** (with one named escape hatch).

- Every tree of a batch — real or placeholder, eager or deferred — is built
  through the single gate `_build_accel`
  (`scene_builder.py:639-672`): `refit` active → `build_refit_bvh`
  (`scene_builder.py:661-663`), else `build_stbvh` (`scene_builder.py:664-672`).
  The docstring states the intent: "All trees of a batch dispatch through this
  one gate so every launch passes a single consistent ``refit`` template"
  (`scene_builder.py:652-654`).
- The gate's live value `refit_bvh_active()` is
  `BVH_REFIT and not WF_TEXTURED and WAVEFRONT_SORT_MATERIALS is not True`
  (`settings.py:849-853`). With defaults all three clauses are true:
  `BVH_REFIT` ON (`settings.py:448`), `WF_TEXTURED` is hard-wired False and its
  setter refuses to enable it (`settings.py:2195`, `2198-2207`),
  `WAVEFRONT_SORT_MATERIALS` starts as string `"0"` (`settings.py:2463`) and its
  setter converts a True request to `"auto"` + raise, so it can never hold
  Python `True` (`settings.py:2466-2476`) — both non-refit causes are dead code.
- Covered build sites: eager full+opaque trees for both geometry types
  (`scene_builder.py:797-819`, `827-849`), absent-type placeholders
  `_empty_scene_part` → `_build_accel(..., refit=None)` → live toggle
  (`scene_builder.py:675-681`), deferred batches record the kind at merge time
  (`scene_builder.py:787`) and `build_deferred_bvhs` forces exactly that kind
  (`scene_builder.py:863`, `852-932`), so a toggle flipped mid-render cannot mix
  tree kinds in one batch.
- `build_stbvh` has exactly one caller in `algan/`: `scene_builder.py:664`
  (grep over the package). So on any live configuration the only way a classic
  STBVH gets built is **`ALGAN_BVH_REFIT=0`** (`settings.py:442-448` documents
  it as "restores the classic per-batch STBVH instance trees"). That is the
  exception to name. The structural exceptions that still exist in the gate —
  legacy textured / sorted-material orchestrators ("``refit_bvh_active`` gates
  them out", `settings.py:850-852`; `DESIGN_hybrid_raster.md:704-705`) — cannot
  be enabled anymore (setters above).

## Claim 2 — six trees; BVH_DEFER skips all of them

**PARTIAL.** The deferral half is CONFIRMED; the "six trees / 3 geometry types"
count is REFUTED as stale.

- Geometry types today are **two**, not three: the merge handles only
  `RayTracedTrianglePrimitive` and `RayTracedBezierCircuitPrimitive` and raises
  `TypeError` on anything else (`scene_builder.py:1279-1286`). The third type
  (PN) is gone — "nothing ever rebound ``RENDERER_REGISTRY.triangle_primitive``
  ... so ``num_pn`` was always 0"; logical PN reaches the renderer pre-diced to
  flat triangles (`DESIGN_hybrid_raster.md:553-560`;
  `DESIGN_mesh_identity.md:737`). A batch therefore builds at most **four**
  trees: `tri_bvh`/`tri_opaque_bvh` and `bez_bvh`/`bez_opaque_bvh`
  (`scene_builder.py:797-849`), with absent/degenerate slots filled by
  placeholders or aliased to the main tree (`scene_builder.py:806-809`,
  `836-839`, `1676-1677`). `DESIGN_hybrid_raster.md:701-702` still says "All
  six trees of a batch (3 full + 3 opaque-prepass)" — written before the PN
  deletion.
- DEFER half CONFIRMED. When `_bvh_deferral_eligible` holds
  (`scene_builder.py:713-746`), `_finalize_bvhs` assigns **one shared
  placeholder object to all four slots** — `tri_bvh`, `tri_opaque_bvh`,
  `bez_bvh`, `bez_opaque_bvh` — and records `bvh_deferred`
  (`scene_builder.py:767-788`); zero real trees are built at merge time. The
  eligibility conditions (`scene_builder.py:727-746`):
  - `BVH_DEFER and HYBRID_RASTER` (`:727`);
  - not `SAMPLES_PER_PIXEL > 1`, not `SHADOWS`, not `INPLACE_AA` (`:729`);
  - not textured, no mem-trim, no custom fragment pipeline (`:731-738`);
  - no `has_refractive` / `has_refl_transparent` /
    `tri_has_reflective`/`bez_has_reflective` (`:739-745`);
  - and the batch must have triangles or circuits at all (`:746`).
  Note the claim lists only "no shadow ray, no reflection/refraction, no
  SPP>1"; the shipped gate is stricter (also INPLACE_AA, textures, mem-trim,
  user pipelines, and HYBRID_RASTER itself). The conclusion survives: the
  common shadow-free non-reflective batch builds zero trees, keeping placeholder
  ABI so kernel launches are unchanged (`settings.py:459-468`).
- On-demand materialization: `build_deferred_bvhs` (`scene_builder.py:852-932`)
  is invoked by the tracer when (a) the batch heads to Monte Carlo,
  `tracer.py:1291-1294`; (b) shadows are on **or** the batch fell back to
  classic primary traversal, `tracer.py:2434-2440`; (c) an actual continuation
  ray was spawned into the secondary drain, `tracer.py:3005-3006`. So the
  rendered output equals the eager build's (`settings.py:463-468`).

## Claim 3 — serial per-type walks; what the default shadow query does

**CONFIRMED.**

- One tree per geometry type present, walked serially, no combined mixed-type
  tree:
  - nearest-hit: `_nearest_surface_g` walks `_nearest_triangle_hit`
    (`raytrace_kernels_taichi.py:1856-1862`) then `_nearest_bezier_hit`
    (`:1869-1877`), each a complete independent traversal, and merges the two
    winners afterwards (`:1879-1897`). Precision note: when the triangle walk
    found a hit, the bezier walk's node window tightens to
    `tt + DEPTH_TIE_EPSILON` (`:1870-1872`) — it is still a separate full walk,
    just pruned.
  - gather: `_collect_hits` — "in one traversal of EACH BVH. Triangles are
    traversed first; the bezier traversal then prunes against the hits already
    gathered" (`raytrace_kernels_taichi.py:2121-2124`).
  - any-hit: `_shadow_anyhit_opaque` — "Trees are tried triangle -> bezier, the
    second skipped entirely on a hit in the first"
    (`raytrace_kernels_taichi.py:2716-2738`).
  - Which types participate is compile-time (`has_tri`/`has_bez` templates)
    from batch contents (`tracer.py:1297-1303`,
    `raster_pipeline.py:2278-2279`; absent types' placeholder trees are skipped
    outright, `raytrace_kernels_taichi.py:2126-2128`). A mixed-type tree is
    explicitly still-planned future work (`DESIGN_hybrid_raster.md:714-715`;
    its "three trees" wording predates the PN deletion).
- Per (event, light) under the default `SHADOW_ANYHIT=False`
  (`settings.py:620-624`; host leaves `shadow_flag = 1`, `tracer.py:1357`, and
  the mode-select at `tracer.py:1368` does not fire; the flag is passed as the
  `anyhit` template at `raster_pipeline.py:2284`): the trace performs the
  **full ordered transmittance march** — `_shadow_occluded` falls through to
  `_shadow_march_occluded` (`raytrace_kernels_taichi.py:2813-2837`, mode 1 adds
  nothing) — which is a peel loop (`while step < MAX_SURFACES_PER_RAY`,
  `MAX_SURFACES_PER_RAY = 256` at `:134`, loop at `:2915`), each iteration a
  fresh **nearest-hit** query via `_nearest_surface_g` (`:2925-2937`),
  multiplying RGB transmittance by the peeled surface's coverage/pass-through
  (`:2989-2999`) until the ray escapes (`found == 0 or t_hit >= max_t`,
  `:2938-2939`) or an opaque blocker retires it (`:3000-3001`). So: neither a
  first-hit early exit nor a single-walk march — it is a transmittance march
  *realized as repeated nearest-hit restarts*, typically one traversal pair per
  peel (one pair total for lit rays and opaque-blocked rays; k+1 pairs for a
  k-surface translucent stack).
- Under `SHADOW_ANYHIT=True` (`tracer.py:1368-1387`): mode **3** (batch
  provably without translucent geometry) compiles the march out entirely and
  runs ONLY `_shadow_anyhit_opaque` — a binary unordered any-hit over
  interval-opaque leaves with first-hit early exit
  (`raytrace_kernels_taichi.py:2800-2812`, walk at `2461-2567`/`2571+`,
  near-first descent, "exits on the first accepted hit", `:2472-2474`); mode
  **2** (translucent but non-transmissive batches) marches normally and, after
  the FIRST partially-transparent peel, spends at most one extra opaque any-hit
  walk over the remaining range, retiring the ray on a hit
  (`:3002-3027`); a **transmissive** batch keeps the plain march because only
  it evaluates pass-through attenuation (`tracer.py:1371-1380`).
- Under `SHADOW_ANYHIT="gather"`: mode **4** replaces the march with
  `_shadow_gather_occluded` (`raytrace_kernels_taichi.py:2814-2825`, body
  `3033-3227`) — the same ordered peel rebuilt on the KBUF gather
  (`KBUF = 4`, `:353`), costing `ceil((k+1)/KBUF)` traversals instead of k+1
  while an all-opaque blocked ray stays at one (`:3064-3073`).

## Claim 4 — topology rebuilt every frame batch; no cross-batch cache

**CONFIRMED.**

- Each render batch window fetches/materializes fresh primitive objects:
  `fetch_batch` → `get_batch_of_primitives` (`render_loop.py:2698-2704`,
  `2005-2027`), which calls `actor.get_render_primitives()` per actor
  (`render_loop.py:2149`) plus the deferred surface/bezier builders
  (`:2156`, `:2199`); prefix slicing likewise produces new objects via
  `slice_time_window` (`:595-602`). New objects start without merge caches.
- The only merge cache is intra-batch: `_merge_scene` returns
  `primitives[0]._rt_merged_scene` if present (`scene_builder.py:1250-1253`),
  set at the end of that same merge (`:1977`); `prewarm_merge_cache` exists to
  fill exactly that cache on the prefetch worker for the *same* batch
  (`scene_builder.py:2063-2081`; `render_loop.py:2705-2725`), and the host-side
  prepared-scene cache is keyed on the same `primitives[0]`
  (`render_loop.py:457-462`). Nothing keys a cached tree on actor-set identity
  or motion; there is no persistence of tree tensors beyond the batch's own
  lifetime.
- Cache invalidation confirms the scoping: dropped on projection/merge OOM
  (`render_loop.py:960-961`, `1013-1014`), after arena upload on the GPU-merge
  path (`:1320-1332`), and on every batch retry/release path
  (`:2885-2887`, `:2934-2936`, `:3045-3047`, `:3078-3080`).
- Consequently `_finalize_bvhs` — the sole place BVHs are built or deferred
  ("Triangle STBVHs are built (or deferred) in _finalize_bvhs once every
  routing flag this batch needs is known", `scene_builder.py:1657-1658`) —
  runs from scratch inside every batch's merge, even when the actor set and
  their motion are identical to the previous window: the inputs are per-window
  frame bounds (`lo/hi` concatenated from the freshly projected primitives,
  `scene_builder.py:1503-1506`, `1852-1853`), and both tree kinds encode the
  window (classic instances segment the window's frames; refit stores
  per-frame links/bounds). Nothing reuses the previous batch's blocks.

## Claim 5 — per-(frame, child) link words scale memory with T × blocks

**CONFIRMED**, numbers below.

- Layout: `blocks [Tb * num_blocks, 8, BVH_ARITY]` — frame t's sibling block
  for internal node i is row `t * num_blocks + i` (`refit_bvh.py:31-37`,
  packing at `:492-513`); kernel side derives the row base as
  `(f % Tb) * num_blocks` (`_refit_row0`, `raytrace_kernels_taichi.py:821-828`).
  Each block carries one int32 link word per child
  (`link = torch.empty((Tb, B, ARITY), int32)`, `refit_bvh.py:442`), packed
  bit-cast into lane-row 6 (f32) or as low/high u16 halves in rows 6/7 (f16)
  (`refit_bvh.py:498-513`; decode `_refit_link`,
  `raytrace_kernels_taichi.py:832-849`). The words are genuinely per-frame:
  `-1` marks a child whose subtree is invisible at that frame
  (`refit_bvh.py:470-489`) and bit 30 is the primitive's *per-frame* opacity
  (`:472-474`, encoding `:44-48`). Because whole block rows are replicated per
  frame, tree memory scales as Tb × num_blocks — the links are what make
  per-frame rows mandatory even where bounds alone might collapse.
- **Bytes per (frame, block)**: 8 rows × BVH_ARITY(4) lanes
  (`stbvh.py:75`). Default `BLOCK_F16=True` (`stbvh.py:96`): 64 B
  (48 B conservative-rounded bounds + 16 B link word halves). f32 variant
  (`ALGAN_BVH_BLOCK_F16=0`): 128 B (96 + 16). The link payload itself is
  ARITY × int32 = **16 B per (frame, block)** in either encoding. Fixed
  stubs (`nodes[1,8]`, three 1-element arrays, `refit_bvh.py:134-137`) are
  negligible and not per-frame; memory accounting sees `blocks` via the
  inherited `get_memory_used` (`stbvh.py:253-263`).
- **Static collapse to T=1: yes, shipped.** Mechanism (upstream of the
  builder): static primitives natively pack single-frame bounds — triangle
  corners carry the primitive's own time dim and bounds are
  `corners.amin(-2)/amax(-2)` over it (`primitives.py:1202-1207`; circuits
  `primitives.py:3382-3383`) — and the merge unifies collections to
  T = max across them (`_cat_collections` → `_unify_time` → `_expand_frames`,
  `algan/rendering/raytracing/utils.py:11-14, 37-60`); `build_refit_bvh` then
  sets `Tb = Tc` (`refit_bvh.py:291`, `:424`) and demands
  `Tc ∈ {1, num_frames}` (`:295-298`). Documented as "static geometry
  (single-frame input bounds) dedupes to ``T = 1``" (`refit_bvh.py:22-26`),
  "`Tc`` may be 1 for static geometry" (`:286-288`), and "``Tb`` is 1 for
  static geometry" kernel-side (`raytrace_kernels_taichi.py:823-824`), matching
  DESIGN §9 (`DESIGN_hybrid_raster.md:689`).
- **Failure condition**: one moving mob. Rigid motion lives in `tri_pos`,
  deliberately not collapsed (`scene_builder.py:1632-1638`), and `_dedup_time`
  is never applied to the bound arrays `lo/hi` — only to
  norm/mat/colors/extra/uvs (`scene_builder.py:1639-1648`, bez analog
  `:1836`). So if ANY primitive animates across the window, its packed bounds
  carry T = num_frames frames, `_cat_collections` expands every other
  collection to that T (`utils.py:41-48`), and `Tb = num_frames` for ALL of
  the batch's trees — including the ones whose members never move. The builder
  itself does not collapse identical frames.

---

## Question A — cost of one shadow-visibility evaluation; multiplying loops

Exact loop nests in the sheet route's shadow trace, `raster_shadow_trace`
(`raster_taichi.py:2744-3045`):

1. `(event × light)` grid, flattened into the launch range:
   `for idx in range(num_events * num_lights)` with `e = idx // num_lights`,
   `li = idx - e*num_lights` (`raster_taichi.py:2821-2823`). Flattening only
   redistributes parallelism; the walk count is events × lights
   (comment `:2813-2820`).
2. Fan-sample loop per cell: `for s in range(ns)` (`raster_taichi.py:2964`)
   where `ns = 1` for hard lights (`:2933`), `ns = SOFT_SHADOW_SAMPLES`
   (default 8) for radius>0 lights (`:2936-2937`), and
   `ns = max(ns, 4)` under `ANALYTIC_AA_SECONDARY_SAMPLES > 1`
   (`:2956-2960`).
3. Occlusion query per sample: `_shadow_occluded(...)`
   (`raster_taichi.py:3021-3036`). Under the default mode 1 this is
   `_shadow_march_occluded`: peel loop `while step < MAX_SURFACES_PER_RAY`
   (=256) (`raytrace_kernels_taichi.py:2915`, cap `:134`), one
   `_nearest_surface_g` call per peel (`:2925-2937`).
4. Inside every `_nearest_surface_g`: one walk per present geometry type —
   `_nearest_triangle_hit` then `_nearest_bezier_hit`
   (`raytrace_kernels_taichi.py:1856-1877`).

So the multiplications are: **events × lights × SOFT_SHADOW_SAMPLES ×
(peels+1) × 2 trees** (hard light: fan=1; lit rays and first-opaque-blocked
rays: peels=1; translucent stacks pay per surface up to the 256 cap). The same
query shape repeats in the deterministic wavefront shade for continuation rays:
hard shadows loop lights serially in-kernel, `for li in range(num_lights)`
(`wavefront_kernels_taichi.py:2305`) calling `_shadow_occluded` per light
(`:2324`), soft fans use `ns = SOFT_SHADOW_SAMPLES` (`:2861`) calling it per
sample (`:2964`). Structural consequence for ranking: the per-peel restart
(item 3→4) and the per-type split (item 4) are constant factors paid on every
one of those axes; §13 item 3 addresses only the last one, and only for
any-hit queries.

## Question B — §13 item 3: a single mixed-type any-hit tree

What its leaves would have to carry (from the current walkers):

- A geometry-type discriminator plus a per-type primitive index: triangles
  index `tri_pos` rows (Möller–Trumbore on 9 floats,
  `raytrace_kernels_taichi.py:2544-2553`); circuits index `circuit_meta` /
  `edges_2d` / `edge_accel` and need `pixel_size_per_t`/`base_dist`, plane
  u/v and the fill/border region test (`_anyhit_opaque_bez`,
  `:2571-2585+`). An any-hit tree needs no returned barycentrics, but the
  intersection predicate itself still branches per type, so the leaf must say
  which arm to run.
- The flag bits the shadow walks already consume from leaf words: the
  interval/full-opacity flag (classic `leaf_tspan` bit 31, `stbvh.py:207-209`
  and `:854-859`; refit link bit 30, `refit_bvh.py:44-48`, `:99`) and the
  no-cast flag (bit 15 classic, `stbvh.py:631-639`; bit 29 refit,
  `refit_bvh.py:100-109`, decode `raytrace_kernels_taichi.py:852-866`). In a
  refit-style word the primitive field is already narrowed to 29 bits to fit
  nocast (`refit_bvh.py:104-109`), so a type bit has to steal one more bit or
  ride an auxiliary array; alternatively a unified primitive index space with a
  remap table.
- Call sites that would consume it — everything that currently walks the two
  trees serially for a boolean answer:
  `_shadow_anyhit_opaque` (`raytrace_kernels_taichi.py:2701-2739`), called from
  `_shadow_occluded`'s mode-3 arm (`:2800-2812`) and mode-2 deferred check
  (`:3017-3025`); and `_shadow_occluded` itself, invoked from
  `raster_shadow_trace` (`raster_taichi.py:3021`) and the deterministic
  wavefront shade's hard (`wavefront_kernels_taichi.py:2324`) and soft
  (`:2964`) shadow paths. The Monte Carlo megakernels would NOT consume it:
  they need ordered hits (`_transmittance`,
  `raytrace_kernels_taichi.py:3488+`; `_nearest_surface`, `:2072+`), so they
  keep per-type trees. Precedent that the merge is tractable:
  `_collect_hits` already interleaves both types into one KBUF buffer with a
  packed hit_type in `hit_flags` bits 0-1 (`raytrace_kernels_taichi.py:2102-2135`)
  — item 3 essentially promotes that gather-level merge into the structure.

## Question C — visibility/traversal work recomputed but shareable

- **Per-peel restart (the big one, intra-ray).** The default march begins a
  complete two-tree nearest-hit traversal for every peeled surface
  (`raytrace_kernels_taichi.py:2915` + `:2926`; stated plainly in the gather
  variant's docstring: "restarts a full three-tree traversal per peeled
  surface", `:3066-3068` — wording predates the PN deletion). A k-translucent
  stack pays k+1 full traversals where the shipped-but-off gather mode pays
  ceil((k+1)/KBUF) (`:3070-3073`); all-opaque blocked and fully lit rays
  already stop after one peel (`:2938-2939`, `:3000-3001`), so the redundancy
  concentrates on translucent stacks and on mode-2-style mixed scenes. This is
  shareable *within* one ray; the gather mode is exactly that share, off by
  default (`SHADOW_ANYHIT=False`).
- **Across (event, light) cells: nothing is shared, by construction.** Each
  cell initializes all state and traces independently; the comment notes the
  per-event setup is recomputed per cell but is pure loads next to a march
  (`raster_taichi.py:2813-2820`). Fan samples share origin/basis setup only.
  Two sheets at the same pixel are distinct events (per-sheet event identity,
  `DESIGN_sheet_resolve.md:717-728`) and legitimately get their own origins.
- **Already-deduplicated upstream** (i.e., NOT recomputed today): the event
  pass accepts only shading points the resolve will actually light, mirroring
  seam de-dup and transport decisions (`DESIGN_hybrid_raster.md:495-499`,
  `DESIGN_sheet_resolve.md:717-728`); zero-colour light rows and geometric
  zero-radiance cases skip tracing entirely
  (`raster_taichi.py:2859-2866`, `:2925-2932`); horizon-culled samples don't
  trace (`:3014-3019`); visibility is computed once per accepted
  (event, light) into the dense `shadow_vis[e, li]` table and read back by the
  resolve's mode-2 pass (`raster_pipeline.py:2202-2210`).
- **Opaque-prepass trees vs full trees: no overlap today.** By default the
  dedicated opaque trees are not built at all — `OPAQUE_BVH_SKIP_DEAD=True`
  aliases `tri_opaque_bvh = tri_bvh` (and bez) whenever no rollout consumes
  them, saving "~40% of the per-batch BVH build"
  (`scene_builder.py:754-766`, aliasing at `:806-809`/`:836-839`;
  `WF_OPAQUE_CLOSEST`/`WF_OPAQUE_PREPASS` default False, `settings.py:126-127`,
  skip default True `:837`). If those rollouts were enabled, the second tree
  would duplicate a subset of the full tree's coverage by design — an opt-in
  trade, not current waste.
- Deferred-BVH batches that turn out to need trees pay one late build instead
  of zero (`tracer.py:2434-2440`, `:3005-3006`) — a latency hedge, not repeated
  work.

---

## What I did not verify

- Nothing was executed: no renders, no pytest, no benchmarks, no kernel
  compilations, no timing of any kind (read-only audit, CPU-only container).
  All verdicts rest on reading the cited source; no wall-clock or device
  performance claims are made anywhere above.
- Compiled-kernel reality: I did not verify which template variants
  (`refit`, `anyhit`, `has_tri/has_bez`, `sec_aa`, `shadow_term`) actually
  instantiate in a given run, nor that no other module bypasses `_build_accel`
  at runtime (static grep found only `scene_builder.py:661/664` as builder
  call sites).
- Sheet-event duplication corner: I did not trace sheet fusion/seam handling
  far enough to prove two accepted events can never share an identical
  (origin, light) cell; I verified only the documented acceptance/dedup
  contract (`DESIGN_hybrid_raster.md:495-499`).
- Batch lifecycle edges: I read the normal fetch/merge/retry paths
  (`render_loop.py:2698-2731`, `2885-3080`) but did not exhaustively rule out
  an exotic path that renders two different windows against one cached merge
  without clearing `_rt_*` caches (e.g., interactions of
  `ALGAN_REUSE_FETCHED_BATCH`/prefetch with mid-batch setting flips beyond the
  deferred-build re-checks cited).
- Opaque-tree consumers: I confirmed the defaults leave `*_opaque_bvh`
  unwalked/aliased, but did not enumerate every consumer under the opt-in
  `WF_OPAQUE_CLOSEST`/`WF_OPAQUE_PREPASS` rollouts.
- `SOFT_SHADOW_SAMPLES` is an initialization-only env read
  (`algan/settings/_startup.py:81`); I did not verify daemon/warm-process
  behavior around changing it (per CLAUDE.md it is set before import).
- Design-doc staleness was noted where it touches these claims (§9 "three
  trees", §11 default table, "six trees"); I did not audit the docs
  exhaustively for other drift.
