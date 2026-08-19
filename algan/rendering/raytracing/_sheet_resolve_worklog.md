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
- Phase 1 (P1–P2 compaction + harness, band rule): CODE DONE, measured on the
  12 harness cases; six-scene stream stats running. RESULTS (md, CUDA,
  benchmarks/_sheet_phase1_md.log):
  * |sheet-E| ties or improves EVERY case: cylfine 0.0050->0.0011 (-78%),
    sphere 0.0057->0.0040, torus 0.0065->0.0048; on-lattice -> ~0 everywhere.
  * Notches: sphere 24@0.0018->14@0.0012, torus 2->0, cylfine 253@0.0090->
    339@0.0012 — count UP but depth 7.5x DOWN: those pixels' TRUE coverage is
    0.999x (not 1), sheets paint the exact area, the old walk rounded to 1.
    Threshold artifact of the notch counter, verified per-pixel
    (truth 0.99920, sheet 0.99875, frag 0.99966).
  * Band rules are output-indistinguishable (unref-dev identical across
    facing/prim c=2/4/8 on every case incl. torus). Fused counts at the
    torus fold: facing 1773, c=2 686, c=4 984, c=8 1458 — fold tangency
    fusion is irreducible (projection non-injective there). DECISION:
    band_rule=prim, band_c=2 (errs toward benign splits; zero false splits
    on all non-fold cases).
  * S/F compaction ratio 0.03 (cylfine, 33x) to 0.99 (flat quad).
    maxS = 4 at md on all cases; >K(24) = 0. Compaction cost ~10-30ms host
    per ~40k-frag stream per arm (indicative).
  * _aa_line_check ink-wobble gate DEFERRED to Phase 2 (needs a real render
    path; the line-check cases' |sheet-E| columns are the Phase-1 proxy).
- Phase 2 (P3–P6 scan resolve behind flag + oracle): IN PROGRESS.
  DONE: ALGAN_SHEET_RESOLVE setting (env+rt_settings+experimental);
  sheet_resolve_taichi.py kernel (per-pixel bounded loop over sheets, walk's
  material split + continuations, per-sheet corr in cfac, sheet-local rule-B);
  prepare_sparse_raster_coverage(sheet_resolve=) compacts (prim c=2), skips
  one-mesh + run lanes, persists sheet arrays, pins "sheets" in dict;
  shade_sparse_raster_coverage launches sheet kernel when pinned;
  tracer passes sheet_resolve=SHEET_RESOLVE && shadow_flag==0;
  oracle resolve_pixel_reference() in sheets.py + unit tests;
  benchmarks/_sheet_resolve_parity.py (A/B lossless + A/A + engagement).
  PARITY RESULTS (LD, benchmarks/_sheet_parity_ld.log): engagement 5
  launches ON / 0 OFF; A/A byte-identical BOTH arms (max|d| 0 — the sheet
  path is deterministic); off-vs-on max|d| 65 over 0.56% of pixels; worst-
  frame panel (algan_outputs/sheet_parity/worst_diff.png) shows movement
  confined to diced-mesh silhouettes/rims + faint interior dusting on the
  lit sphere (per-sheet dominant-fragment shading) — the intended change,
  no notches/seams/structure.
  Fast suite flag-off: 213 passed (default path untouched).
  Oracle-vs-kernel verify: PASS, worst |claim| diff 8.94e-08 over 61
  committed sheet rows at 24 probe pixels (matte scene, dump-fed alphas).
  WOBBLE A/B TRAP HIT AND FIXED: the first A/B came back IDENTICAL in every
  row — a warm render DAEMON (auto-started by an earlier harness run) served
  BOTH arms with its own environment, so ALGAN_SHEET_RESOLVE never reached
  the render. The known "live daemon serves STALE code" trap in env-var
  costume. Daemon killed; re-running both arms with ALGAN_USE_DAEMON=0 and
  ALGAN_AUTO_DAEMON=0. Rule for every env-var A/B in this project: disable
  the daemon in BOTH arms or the arms are one render.
  FIRST DAEMON-FREE WOBBLE A/B: coarse cyl REGRESSED 2-4x (far-sheet
  re-claim; §7's cap-subsumption claim refuted by measurement). FIX: the
  one-mesh ceiling restored AS SHEET DATA — host reduction kept, sheet_cap
  on the record, bounded-loop clamp in kernel + oracle (occlusion scaled
  with claim). RE-VALIDATED (all green):
  * parity: engagement ✓, A/A max|d| 0 both arms, off-vs-on 7479 px.
  * oracle verify: PASS 8.94e-08 / 56 rows / 24 pixels.
  * wobble: bez/quad identical; cyl mean 0.0159->0.0050 (-69%); cyl_fine
    0.0442->0.0130 (-71%), worst 0.0878->0.0165; only the two degenerate
    axis-aligned rod angles +<=0.001.
  Phase 2 gates MET. Committing.
  NOTE-TO-SELF: killed the first parity run after editing the kernel file
  mid-run (JIT source rule); orphans killed, clean rerun done.
- Phase 3 (unification): CODE DONE, validating. Implemented:
  * sheet_route decided ONCE in render_batch_raytraced (SHEET_RESOLVE &&
    shadow_flag==0 && analytic_raster && ANALYTIC_AA_RUN && samples<=1 &&
    !transparent_background), passed down to raytrace_render_wavefront.
  * env_background_prefill kernel (sheet_resolve_taichi): env(ray)*255 per
    (frame,pixel) into `out` in render_chunk when env_meta && sheet_route.
  * sheet kernel env_in_composite template: retire hands leftover weight to
    the composite instead of folding env (prefill IS the primary ray's env).
  * sparse gate: `(not env_active and t_val==3) or (sheet_route and
    use_raster)`. prepare(require_sheets=) raises on emission refusal when
    the relaxation was load-bearing.
  * wf_composite_accum_sparse gained tonemapping template (3 = old exact
    path); new wf_finalize_uncovered(mask) pays finalize(bg) for untouched
    pixels under in-kernel tonemap, mask scattered from covered_idx, runs
    even when coverage is None (empty env/tm frames).
  RUNNING: parity --scene env and --scene tm (A/B + A/A, engagement).
  Old plan notes below.
  PLAN (sketched while Phase-2 runs):
  * Core deliverable: the sparse+sheets route serves ENV-MAPPED and
    IN-KERNEL-TONEMAP batches, so the sparse_coverage gate drops its
    `not env_active` and `t_val == 3` conditions when SHEET_RESOLVE is on.
  * Env (mechanics CONFIRMED by reading): the sparse route prefills the
    frame buffer `out` with the background (`_prefill_background`) and
    `wf_composite_accum_sparse` writes `out[p] = acc*255 + weight *
    out_prefill[p]` for covered pixels only. So background-as-final-sheet
    = (a) an `env_background_prefill` kernel writing env(ray)*255*
    intensity into `out` per (frame, pixel) instead of the flat color, and
    (b) the sheet kernel NOT folding env at retire on this route (pass
    env cols zeroed in layer_offsets) so the leftover weight multiplies
    the env-prefilled out in the composite. Bounced rays keep sampling
    env at miss inside wavefront_shade (their pixel's bg would be wrong).
    Empty pixels then need NOTHING — the prefill already holds env.
  * Tonemap: resolve stays linear (it already is); with POST_PROCESS_
    TONEMAP=0 the composite must run un-compacted with in-kernel tonemap
    (the dense composite variant already exists). Launch policy, not
    resolve semantics.
  * Dense path stays as the fragment walk (kill-switch role) until Phase
    4 decides; on the sheet route it simply stops being selected.
  * Gate: the Phase-0 env_*/tm_* order-window arms must render through
    sheets and stay lever-inert; env noise floor 1 -> expected 0 for
    covered pixels? (pix_accum atomic remains for bounce adds — the full
    §2.2 zero-floor claim lands with P7 in Phase 4.)
- Phase 4 (P7, §H, §I, deletion, re-baseline): NOT STARTED. SEQUENCING PLAN:
  4a. Shadow events from SHEET records: event id = SHEET INDEX (dense
      per-sheet tables, zero bookkeeping, deterministic by construction — no
      count pass, no atomic reserve). One shared @ti.func for the per-sheet
      claim arithmetic serves resolve + event build (kills the lockstep
      drift class). Then drop the shadow_flag==0 restriction on the route.
  4b. P7 deterministic continuation slots for iteration 0: resolve runs a
      COUNT pass (same kernel, count template — resolve is a few % of
      frame, 2x is affordable), host int-scan -> per-pixel base, emit pass
      writes base+ordinal. No rs_alloc atomic in iteration 0. Measure §J's
      MD noise floor (46 -> predicted drop; the wavefront loop's own deeper
      splits keep their allocator, out of P7 scope — say so).
  4c. §H nested IOR (rs_sca cols 7+, relative eta at interfaces, overflow ->
      air fallback) in sheet kernel glass branch + wavefront shade; §I
      self-shadow by identity (sheet_sid on the record — compact_sheets must
      output it; accept rule same-mesh-only-at-near-zero-t).
  4d. Flip ALGAN_SHEET_RESOLVE default ON; delete raster_first_shade +
      raster_shadow_event_build + run machinery + ladder (the one-mesh HOST
      reduction and cap stay — Phase 2's measured correction); ONE
      re-baseline (CUDA sets here; CPU sets = new epoch, stated in commit);
      final design-doc + AGENTS_DETAILED/CLAUDE.md updates; delete this
      worklog.
  GLOW-LANE BUG (env prefill): col 3 of the frame buffer is the GLOW lane;
  writing 255 bloomed every pixel white (max|d| 222 over the whole frame).
  Fixed to 0 (the sky emits none — matches the dense arm's retire).
  4a STATUS: kernel mode templates (0 plain / 1 event build / 2 shade w/
  vis) in ONE body; host three-mode launch + compact + raster_shadow_trace
  reuse; sheet_route no longer requires shadow_flag==0. Basic parity
  unchanged; shadow parity found a BRIGHT SEAM through the translucent
  quad = a FUSED sheet (quad + backdrop BOTH sid 3, gap 1.04, my band
  scale used raw triangle extent — the huge backdrop's ~4-unit extent
  swamped the gap). FIX: band scale = per-PIXEL depth slope (extent /
  projected px size from tri_screen, valid-flag guarded) + pws*t; unit
  test pins it; spy confirms the split. NOTE the identity finding for
  later: two DIFFERENT generic triangle mobs (TriangleTriangulated,
  QuadTriangulated) shared one sid — the old run rule had the same
  exposure (measured small); worth a mesh_key on generic tri mobs in the
  §4.1 identity sweep. Shadow parity re-running after the fix.

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

## Phase 2 architecture (settled while Phase-1 measurements ran)
- Setting: ALGAN_SHEET_RESOLVE (default 0), environment.py + experimental.
- Engages on the SPARSE path only, and only when shadow_flag == 0 (shadowed
  batches keep the old walk until Phase 4 builds shadow events from sheet
  records, per the design's own phasing). Flag off = old walk untouched =
  byte-identical trivially.
- prepare_sparse_raster_coverage, when flag on: skip the one-mesh reduction
  and run lanes (machinery the sheet path deletes), run compact_sheets
  (prim, c=2), return sheet arrays in the coverage dict (persist arena).
- ONE new kernel `sheet_resolve_shade` in raster_taichi.py (shares all
  shading helpers): one thread per covered pixel, bounded loop over its
  sheets. Per-sheet semantics:
  * circuits: areal, exactly the walk's (alpha *= SDF cov).
  * tri, union == 0 (donor sheet): areal a[s] = alpha*min(cov,1) — replaces
    old run_mode 2's sequential renormalization entirely (one record).
  * tri, union == ALL: corr = 1 if |1-cov| <= _AA_FULL_DUST else min(cov,1);
    a[s] = alpha*corr on all samples.
  * tri, partial union: corr = min(cov,1)/Q; a[s] = alpha*corr on owned
    samples; corr>1 clamps with SHEET-LOCAL rule-B redistribution onto
    unowned samples (per-record arithmetic, no cross-record state).
  * NO one-mesh cap, NO run scan, NO seam dedup, NO engagement gate.
  * material 4-way split / continuations / sec_aa / glossy identical to the
    walk, evaluated ONCE per sheet at the dominant fragment.
  * P5-as-separate-material-sorted-kernel deferred (optimization, not
    semantics; single per-pixel kernel first). Sibling machinery (§4.4)
    currently vacuous: compaction emits exactly one record per band.
- Oracle: sequential_reference() beside compact_sheets (host python),
  matte+alpha first; parity harness _sheet_resolve_parity.py renders
  flag off/on lossless + diffs, and oracle-vs-kernel via ALGAN_AA_DUMP
  extended to the sheet kernel.
- KNOWN semantic deltas vs old walk to measure in parity (expected, will
  move output): far-sheet re-claim residual (cap deleted) ~(Q-area)*corr at
  silhouettes; notch population becomes exact-area (see Phase-1 finding);
  split/capped populations gone.

## Next action
Commit Phase 1; read six-scene stats when done; then build Phase 2
(setting + prepare wiring + sheet_resolve_shade kernel + oracle + parity).
