# Sheet resolve — session worklog (delete before final merge)

Working file for the sheet-resolve implementation session started 2026-08-19.
`DESIGN_sheet_resolve.md` is the plan of record; this file tracks execution
state so the session can resume from summarized context. Keep it terse and
current. Delete it in Phase 4's cleanup commit.

## Machine facts (do not re-derive)
- Windows, GTX 1050 4GB, owns CUDA baselines; must NOT write CPU baselines
  until Phase 4's deliberate re-baseline (which regenerates both, new epoch).
- Wall-clock A/B unusable (thermal throttle). Use counts, byte-diffs,
  in-process alternating A/B with a control.
- Cold Taichi rebuild ~10 min for harness scene sets; new ti.static template
  VALUE = new variant = new compile. Do not wipe ~/.algan/cache wholesale —
  back up/restore `~/.algan/cache/taichi` only, or point
  TI_OFFLINE_CACHE_FILE_PATH at scratch.
- One render process at a time. Never edit *_taichi.py during a render.
- ruff: `--no-fix` always; never name *_taichi.py on a ruff command line.
- Lossless A/B: codec="libx264rgb", ffmpeg_params=["-crf","0"].
- venv: `.venv\Scripts\python.exe`.

## Phase status
- Phase 0 (instruments): DONE (2026-08-19). _order_window_check.py renders
  lossless + env_*/tm_* arms with per-family noise floors. Validated: env
  floor 1 (pix_accum atomic cap), tm floor 0, at LD on CUDA. Full lever
  sweep on the extended arms deferred to phase gates (each arm is a
  subprocess render).
- Phase 1 (P1–P2 compaction + harness, band rule): IN PROGRESS
- Phase 2 (P3–P6 scan resolve behind flag + oracle): NOT STARTED
- Phase 3 (unification): NOT STARTED
- Phase 4 (P7, §H, §I, deletion, re-baseline): NOT STARTED

## Phase 1 design notes (settled before building)
- Compaction lives in NEW engine module algan/rendering/raytracing/sheets.py,
  host torch, deterministic (int ops + f64-accumulate-round for areas — the
  §6.6.4 pattern). Called only by the harness in Phase 1 (no shipped-path
  change). §10.4 leaves scan implementation shape open; host torch is the
  Phase-1 implementation, revisit at Phase 2 for the shipping path.
- Sheet key: (pixel, sid2, band) where sid2 = tri_obj-sid * 2 + facing for
  triangles; each bezier fragment is its own sheet (circuits never group;
  border/fill blend already packed per fragment).
- Band rules to measure: R0 facing-only (no depth split, the fallback);
  R1 local-relative: gap > c * (ext_i + ext_j + t*pixel-angular-size), ext =
  triangle's own camera-distance extent (max_k|v_k-ro| - min_k|v_k-ro|),
  c swept {2,4,8}. Fusion detector = band with a sample bit contributed
  twice (fill-rule partition violation); splits are benign (§6.2).
- Phase-1 scoring: feed one-fragment-per-sheet lists through the VERIFIED
  host replay of the existing walk in _aa_run_gate_check (spy already
  intercepts prepare_sparse_raster_coverage). New column |sheet-E| + sheet
  notch/fused/split counts. Torus case added (concave: two same-facing
  sheets per pixel at the fold).
- IMPORTANT finding while reading the design: §4.3's per-sheet corr leaves a
  bounded far-sheet re-claim at silhouettes (≈ (Q-area)*corr) because the
  one-mesh cap is deleted (§7). §6.1 accepts this class. Phase 1 feeds
  sheets through the EXISTING walk (cap still on via ONE_MESH bit + caps),
  so this is a Phase-2 question; measure it there against the oracle.
- corr > 1 (sub-sample rods): §4.3 formula unclamped is exact on the claim;
  occlusion needs band-local rule-B-style redistribution ("second pass, not
  walk state") — per-band arithmetic, no cross-band feedback. Phase-2
  decision, recorded here so it is not lost.

## Key code facts learned (append as discovered)

### Fragment stream (sparse path) — P0's output, already exists
`prepare_sparse_raster_coverage` (raster_pipeline.py:1144) emits, per batch
window (NOT per tile — whole frame window at once):
- `frag_key` i64: `(chunk-relative pixel << 32) | f32 depth bits`. Pixel index
  is `lp = (f - time_start) * ppf + p` (ppf = width*height, tile_start=0 on
  sparse path). Depth recovered by bit-view (`_frag_t`).
- `frag_ref` i32: triangle prim id >= 0, or `-(circuit << 8 | border_frac) - 1`
  (bezier; `_BEZ_BORDER_BITS = 8`).
- `frag_ab` f32[N,2]: barycentrics (b,c) or plane (u,v).
- `frag_cov` f32: exact clipped area (run repr) or popcount/N; circuits: SDF
  coverage. Pre-filled 1.0.
- `frag_msk` i32: low 8 bits sample mask (D3D 8x pattern, `_AA_NUM_SAMPLES=8`,
  `_AA_MASK_ALL=0xFF`); flags at bit 16+: `_AA_BACKFACE_BIT` (1<<16),
  `_AA_SLIVER_BIT` (2<<16), `_AA_ONE_MESH_BIT` (4<<16).
- `frag_cap` f32: one-mesh ceiling (max sheet area, f64-accumulated), 2.0
  sentinel = no ceiling.
- `frag_run_e`/`frag_run_uw`: §6.7 exact-run lanes (1-elem dummies unless
  ANALYTIC_AA_RUN_EXACT).
- `covered_idx` i32[num_covered]: ascending covered compact-pixel list.
- `run_offsets` i32[num_covered+1]: CSR into frag_* per covered pixel.
- Ordering: `_exact_fragment_order` = stable argsort desc layer, then stable
  argsort (pixel << 32 | depth_bin), depth_bin = floor(t/DEPTH_TIE_EPSILON).
- Opaque truncation: pixel's prefix ends at first fragment that is materially
  opaque AND occludes every sample (full mask + full coverage under RUN_FULL).
- One-mesh reduction + cap + run lanes computed here host-side (scatter ops,
  f64-accumulate for reproducibility).
- Returned dict pins `aa_bez`, `aa_tri`, `aa_grp` at emission.
- mesh id: `sid = tri_obj[_tri_obj_row(pix, ppf, time_start, rows), ref]`;
  `_tri_obj_row = ((pix // ppf) + time_start) % rows`. tri_obj is [T,N] or [1,N].

### The resolve being replaced
`raster_first_shade` (raster_taichi.py:3786): one thread per covered pixel,
serial walk with svis[8] per-sample transmittance, run-rule state machine
(run_end/run_mode/run_corr/mesh_ink/...), 4-way share split
(shade/reflect/transmit/miss), spawns continuations via `_spawn_pool_ray`
(atomic `rs_alloc`), in-place bounce = writes ro/rd + rs_int status ACTIVE and
`break`s (fragments behind a bounce are abandoned — path-bending termination).
Retire: atomic-add acc into `pix_accum[r,0:4]`, leftover weight into cols 4-6;
env map sampled at retire when env_w > 0. `raster_shadow_event_build`
(:2934) replays the same walk to accept lit shading points.
Shading funcs used: `_tri_color_g`, `_tri_extra_g`, `_shade_tri_hit` (frag
pipelines), `_tri_normal_g`, `_tri_ior_transmission_g`,
`_sample_circuit_color_blend`, `_bezier_normal`, `_material_reflectance`,
`_reflect_frame`, `_refract_ray`, `_sample_env_map` (from
wavefront_kernels_taichi / shading_taichi).

### Path selection (Phase 3/5 deletes these conditions)
tracer.py:2096 `use_raster` = HYBRID_RASTER && masks && (tri>0||bez>0) &&
!textured && !mem_trim && no custom scatter && near_clip<=0 && aa_level<=1.
tracer.py:2118 `sparse_coverage` = use_raster && RASTER_SPARSE_COVERAGE &&
prefill && COVERED_SHADE && **not env_active** && **_get_tonemap_t_val()==3**.
Dense fallback = `raster_iteration_zero` (per-tile z-buffer + CSR, same walk
kernels with covered/compact templates differing).
`analytic_raster_route_active` (tracer.py:378) decides AA=1 vs supersample.
aa ladder: `_aa_group` -> aa_grp 0/1/2(RUN_FULL)/3(ONE_MESH)/4(+DENS)/
5(RUN_CAP)/6(RUN_EXACT); `_aa_group_dense` caps at 5.

### Machine/API notes
- Scene.save_video(path, quality, overwrite=True, codec=..., ffmpeg_params=...)
- `SETTINGS.computing.set(available_memory_override=N)` pins batch windows.
- SceneManager.reset() between authored scenes in-process.
- KERNEL_REGISTRY.render_kernel hook counts batches (see _order_window_check).

## Decisions made this session
(none yet)

## Next action
Phase 0: extend _order_window_check.py (env-map scene + non-default-tonemap
scene arms + lossless codec), check tonemap/env API names first.
