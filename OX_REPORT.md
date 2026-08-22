# OX_REPORT — shadow terminator fix on diced surfaces (work-queue item 20)

Branch `claude/shadow-terminator-diced-jn27xq`. Written by Ox Alpha over three
rounds; edited only to add this note.

> **Round 1's "headline finding" below is superseded — it was a scene defect,
> not a code one.** The prescribed test scene gave every mob Algan's default
> material, which renders UNLIT, and an unlit fragment builds no shadow event
> at all, so the scene could not show the feature whatever the code did. With
> an explicit `MeshStandardMaterial` the effect is plain: a lit `Torus` at LD
> goes from 41 speckle pixels to 4, and the diagnostic `relax` arm stays at 38.
> Round 1's reasoning and its instrumentation were sound; only its conclusion
> that the scene was representative was wrong. Rounds 2 and 3 stand as written,
> and round 3's four findings against the shipped diff and its docs were all
> re-derived independently and acted on.

## The headline finding (read first)

**The brief's expectation "`relax` must differ from both" does not hold on the
prescribed scene: `/tmp/term_relax.png` is byte-identical to
`/tmp/term_off.png`** — same size (20476), same md5. This is reported as a
finding per the brief ("do not paper over it"), together with the evidence that
the feature **is** engaging mechanically and that the identity has a specific,
measured cause: on this scene every sample the relaxed guard newly admits lands
on a sheet that contributes nothing visible to its pixel.

Evidence chain for "engages but cannot show here" (all instrumentation since
removed; the shipped code is the clean implementation):

1. Temporary counters inside `raster_shadow_trace` (relax arm): 29903 shadow
   events; 117750 samples evaluated with the relaxed predicate; 110804 admitted
   by it; **699 samples flipped** (admitted by `snrm·wis > 1e-4` but rejected
   by today's `fnrm·wis > 1e-3`). `shadow_vis` changed in **138 cells**
   (checksum sum 89385.664 → 89267.914, partially-lit cells 401 → 539; max
   per-cell |Δvis| = 1.0).
2. Mapping the 138 changed cells to their pixels (via the sheet CSR +
   `covered_idx`) and diffing the PNGs: those 138 pixels — and indeed all
   419904 pixels — are identical between off and relax.
3. Control A: overwriting **exactly those 138 cells** with `vis = 0.123` on the
   host before the mode-2 shading pass → PNG still byte-identical to off.
   So those cells' events belong to sheets whose shading reaches no output
   pixel (self-occluded far-side geometry of the opaque torus / sub-alpha
   sheets).
4. Control B: forcing **every** cell to 0.123 → image changes drastically
   (20476 → 12864 bytes). So the consumption path
   (`shadow_vis` → `sheet_event_id` → `_shade_tri_hit(... shadows, lvis)` →
   pixels) works end to end; the disconnect is not plumbing.
5. Independent confirmation: an untracked A/B harness I did not write appeared
   in the tree during the session (`benchmarks/_shadow_terminator_ab.py`,
   mtime 01:12, after session start). Its own run agrees line by line:
   torus off-vs-on moves (OK, 18 px), sphere band OK (5 px, max|d|=1),
   cube byte-identical OK, circuit byte-identical OK — and its acne assertion
   **FAILS** with speckle counts off=1 on=1 relax=1: there is no visible acne
   in *any* arm of this scene at LD, so there is also none for the offset to
   remove. Its outputs were deleted after the run; the script itself was left
   untouched.

Conclusion: implemented as briefed; verified engaging; the prescribed scene
cannot demonstrate the phenomenon visually at LD on CPU. Item 20's own note is
consistent with this — it warns a convex solid cannot test any of this, but the
same turns out to be true of this concave torus at this dice/resolution: the
light-grazing band's fragments sit where their sheets do not contribute.
Demonstrating the fix visually needs a different scene/camera/light/dice
configuration, which the brief forbids me from authoring.

## What I changed and where

* `algan/rendering/raytracing/shading_taichi.py` — new `@ti.func
  _shadow_terminator_delta(f, prim, w0, a, b, p, snrm, tri_pos, tri_norm)`
  next to `_orient_hit_normals`: Hanika's offset (Ray Tracing Gems II ch. 4).
  Reads the three raw vertex normals out of `tri_norm[f % rows, prim, 0..8]`
  and positions out of `tri_pos` (same layout as `_tri_surface_point`),
  normalizes each (any norm < 1e-9 ⇒ returns the zero vector), picks one sign
  `sgn = +1 if snrm·(w0·n0+a·n1+b·n2) >= 0 else −1` against the caller's
  ORIENTED shading normal (raw interpolated normal used for the test, never a
  normal-mapped result), then computes `d_i = min(0,(p−p_i)·(n_i·sgn))` and
  returns `−(w0 d0 n0' + a d1 n1' + b d2 n2')`.
* `algan/rendering/raytracing/sheet_resolve_taichi.py` — `sheet_resolve_shade`:
  new template param `shadow_term` immediately after `mode`; new ndarray param
  `event_toff` immediately after `event_dp`. In the mode-1 event build, inside
  exactly the block that sets `sheet_accept[idx] = 1`, under
  `ti.static(shadow_term == 1)`, stores the delta into `event_toff[idx, 0..2]`.
* `algan/rendering/raytracing/raster_taichi.py` — `raster_shadow_trace`: new
  ndarray param `event_toff` right after `event_dp`; new template param
  `shadow_term` appended last. `sorigin = spos + fnrm·(10·MIN_HIT_DISTANCE)` is
  kept and the stored delta is added on top when the gate is 1 and any stored
  component is non-zero; `lifted` records that. The horizon guard became
  `horizon_ok = (fnrm·wis > 1e-3) and (snrm·wis > 1e-4)`, replaced by only
  `snrm·wis > 1e-4` when gate ≠ 0 and lifted == 1 — written so the
  gate-off/flat-facet path visibly keeps the original two-sided test.
* `algan/rendering/raytracing/raster_pipeline.py` — `shade_sparse_raster_
  coverage`: reads `term_mode = int(rt_settings.shadow_terminator_mode())`
  live once before the mode split; allocates
  `event_toff = _arena_tensor(memory, (S if term_on else 1, 3), torch.float32)`;
  passes the gate and the array to all three `sheet_resolve_shade` launches
  (mode 0 passes `dummy_f3`); computes
  `ev_toff = event_toff.index_select(0, acc_idx) if term_on else event_toff`
  and hands it plus `term_mode` to `raster_shadow_trace`.
* `algan/rendering/raytracing/wavefront_kernels_taichi.py` — both inline shadow
  blocks (`wavefront_shadow` ~line 2147 area and `wavefront_shade`'s inline
  block ~line 2554 area): apply the helper inline (no array; hit point is
  `spos = ro + t_hit·rd`), same `lifted` bookkeeping, same guard relaxation in
  their light/sample loops. New template param on each kernel.
* `algan/rendering/raytracing/tracer.py` — both `wavefront_shade` launch sites
  pass `int(rt_settings.shadow_terminator_mode())` in the new parameter
  position (after the `deferred_shadows` slot), read live per batch like the
  other shadow toggles.
* Settings registration, following `SHADOW_IDENTITY_REJECT` end to end:
  * `algan/rendering/raytracing/settings.py`: tri-state global
    `SHADOW_TERMINATOR` default ON (`ALGAN_SHADOW_TERMINATOR=relax` or `2`
    selects the diagnostic arm; parsed like `SHADOW_ANYHIT`'s string case,
    junk values warn via `env_flag` and fall back to ON),
    `set_shadow_terminator(enabled)`, `shadow_terminator_mode() -> int`
    (documented why it must be read live and returned as an int: template
    value, offline cache).
  * `algan/settings/raytracing_settings.py`: `"SHADOW_TERMINATOR"` legacy name
    (experimental field, not public) and
    `"shadow_terminator": "set_shadow_terminator"` override.
  * `algan/environment.py`: `"ALGAN_SHADOW_TERMINATOR"` declared in
    `_IMPORT_TIME_VARIABLES` (the tuple holding `ALGAN_SHADOW_IDENTITY_REJECT`),
    alphabetically placed.
* Untouched, as instructed: `raytrace_kernels_taichi.py` (Monte Carlo
  megakernel), every reflection/refraction continuation origin, everything
  under `tests/*/expected_outputs_*`.

## Deviations from the brief

1. **Build-side write gate is `shadow_term == 1`, not §2's `!= 0`.** The brief
   contradicts itself: §2 says compute+write under `!= 0`, §6 says "
   `shadow_term != 0` relaxes the guard, `shadow_term == 1` applies the delta.
   `event_toff` is full-size only in mode 1." Writing under `!= 0` writes into
   a `(1,3)` arena tensor at relax-mode indices up to S−1 — an uninstrumented
   out-of-bounds arena write (I observed this configuration during diagnosis
   before fixing it). §6's semantics win; the code comment states why.
2. **`term_mode` is read before the `if shadow_flag:` split**, not literally
   "next to `identity_on`": all three `sheet_resolve_shade` launches take it as
   a template and `event_toff`'s allocation size depends on it, so it is
   needed upstream of the first launch. One live read, reused everywhere.
3. **`lifted` semantics reconcile §3 and §6**: §3 defines `lifted` = "delta
   non-zero" and relaxes only when lifted; §6 says mode 2 relaxes the guard
   while applying NO offset. In the kernels: modes 0/1 follow §3 literally;
   mode 2 sets `lifted = 1` for every event so the relaxation is unconditional
   there. Commented at each site.
4. **Citation left exactly as the brief gave it** ("Hanika, Ray Tracing Gems II
   ch. 4") — web search was unavailable (403), so I did not guess a chapter
   title beyond what the brief asserts.
5. Note, not a code deviation: the flat-facet "exactly zero / bit-for-bit"
   property is exact in real arithmetic; float evaluation of `d_i` can leave
   ulp-scale dust making `delta` bitwise-non-zero on such facets (which would
   set `lifted=1`). Empirically irrelevant — see the byte-identical cube/circuit
   controls below — but the strict bit-for-bit claim rests on measurement here,
   not construction.

## Verification actually run (exact outputs)

1. `.venv/bin/python -c "import algan"`:
   ```
   Rendering device set to cpu
   [Taichi] version 1.7.4, llvm 15.0.4, commit b4b956fd, linux, python 3.11.15
   [Taichi] Starting on arch=x64
   ```
   Clean — no unknown-`ALGAN_`-variable warning. (Run again after the final
   edits.)
2. `.venv/bin/python -m pytest -q tests/unit_tests/test_environment.py` →
   `19 passed, 3 warnings in 6.36s`.
3. `.venv/bin/python -m pytest -q tests/unit_tests/test_raytracing_unit.py` →
   `7 passed, 4 skipped, 3 warnings in 2.63s`; skips are pre-existing and
   unrelated (`SKIPPED [4] ... deterministic render_scene_stbvh megakernel was
   removed (commit ceaf3c4)...`).
4. Three-arm render from the repo root (exact brief commands):
   ```
   ALGAN_USE_DAEMON=0 ALGAN_SHADOW_TERMINATOR=0     .venv/bin/python /tmp/ox_term_scene.py /tmp/term_off.png
   ALGAN_USE_DAEMON=0 ALGAN_SHADOW_TERMINATOR=1     .venv/bin/python /tmp/ox_term_scene.py /tmp/term_on.png
   ALGAN_USE_DAEMON=0 ALGAN_SHADOW_TERMINATOR=relax .venv/bin/python /tmp/ox_term_scene.py /tmp/term_relax.png
   ```
   ```
   920b24f882a45e751e419f52dd27f6a7  /tmp/term_off.png   (20476 bytes)
   8331828cbb15bb84f9a00c1288eaf15e  /tmp/term_on.png    (20472 bytes)
   920b24f882a45e751e419f52dd27f6a7  /tmp/term_relax.png (20476 bytes)
   ```
   * off vs on: **differ** — 18 differing pixels of 419904, max channel delta
     5 (7 pixels exceed the suite tolerance of 2). All three arms completed
     without exception.
   * off vs relax: **byte-identical — THE FINDING above.**

Additional verification beyond the brief's list:

* Lint: `ruff format --check` passes on every non-`*_taichi` file I touched
  (HEAD was clean; my edits keep it clean). `ruff check --no-fix`: my new code
  introduces zero findings (one SIM114 I introduced was fixed); remaining
  findings in touched files (D209 in `shading_taichi._two_sided_normal`;
  I001/F811/F841 in `tracer.py`/`wavefront_kernels_taichi.py`) exist at HEAD
  and CI runs only `ruff format --check`.
* Flat-facet / bezier guarantees: `benchmarks/_shadow_terminator_ab.py` (not
  mine; see finding §5): cube and circuit arms **byte-identical** across all
  three settings; torus moves between off/on. This is direct measured support
  for the load-bearing "flat-shaded geometry stays byte-identical" claim.
* Classic wavefront path (`ALGAN_HYBRID_RASTER=0`, otherwise the torus scene):
  rendered all three gate values — no Taichi compile/runtime errors, so both
  modified inline blocks compile and execute in all three variants. Outputs
  byte-identical across arms there too (15735 bytes each, md5
  `bcf339ae35a7874b1d8acd42f1b4c6a0` ×3), consistent with the scene-level
  finding.
* Settings surface smoke test: `set_shadow_terminator(False/True/"relax")` →
  modes 0/1/2; `SETTINGS.raytracing.experimental.set(shadow_terminator=...)`
  works; the field round-trips through `to_dict()`.

## What I did NOT verify

* **Anything on CUDA.** This container is CPU-only. Kernel compilation,
  performance and output on a CUDA device are unchecked; per CLAUDE.md a
  kernel change wants a CUDA machine before it ships.
* **`tests/fast` and `tests/full_renders`.** Not part of the brief's
  verification list and deliberately not run; `tests/fast`'s pixel comparison
  fails on master by ~40 channel values against a tolerance of 2 (stale CPU
  baseline, work-queue item 17) — pre-existing, not chased.
* **No human visual inspection.** All comparisons are numeric (md5/pixel
  diffs/speckle metric). Whether the 18 moved pixels look right is unjudged.
* **The Monte Carlo megakernel** (`samples_per_pixel > 1`) is untouched and
  untested with these settings, per the brief.
* **`wavefront_shadow` at runtime.** It has no caller anywhere, so Taichi
  never compiles it; my edit there is Python-syntax-checked (py_compile) and
  mirrors the exercised `wavefront_shade` block line for line, but its kernel
  body itself never executed. `wavefront_shade`'s inline block WAS executed
  (classic-route runs above).
* **Whether the fix achieves item 20's "Done when".** That criterion ("a diced
  curved surface under a grazing light shows no acne with the guard angles
  relaxed") is not demonstrable on the prescribed scene — see the finding.
  Establishing it needs a scene where the band's fragments contribute visibly,
  which I was told not to author.

## Round 2

Three defects fixed, then the unit tests. Nothing from round 1 reverted; work
tree left dirty, nothing committed.

### Defect 1 — out-of-bounds read of `tri_norm` under memory trimming

`_shadow_terminator_delta` (`shading_taichi.py`) indexed
`tri_norm[tn, prim, 0..8]` unconditionally, but on the classic wavefront path
`wavefront_shade` calls it with `mem_trim` live and a `tri_norm` that may be
the compacted needs-normal prefix — shorter in its second dimension than
`tri_pos`. Fixed INSIDE the helper (not at the call sites): the whole body now
sits under `if prim < tri_norm.shape[1]:`, returning the zero vector
otherwise, with the reason in the comment citing
`_flat_triangle_normal_trim`, whose own guard carries the same rationale ("a
bare prim (index past the prefix) never consumes the shading normal"). On an
untrimmed array `shape[1]` is the full triangle count, so one guard covers
every caller; the sheet route passes merged full-size arrays and is unaffected.
Docstring documents the case alongside the degenerate-normal one.

### Defect 2 — the flat-facet guarantee is now by construction

After normalizing, the helper tests the three vertex normals for agreement —
`n0.dot(n1) > 1 - 1e-6` AND `n0.dot(n2) > 1 - 1e-6` — and skips the formula
outright when they agree. The comment states this is the DEFINITION of the
constant-normal-field case (a facet with one has no smooth surface to be
displaced onto), not a tolerance hack: float evaluation of `d_i` could leave
ulp-scale dust making `delta` bitwise non-zero, setting `lifted = 1` and
relaxing the horizon cull on geometry that never moved. Updated every prose
site that claimed exactness as an arithmetic consequence so it cites the
equality test instead:

* `_shadow_terminator_delta`'s docstring (`shading_taichi.py`);
* the `SHADOW_TERMINATOR` settings block (`settings.py`);
* both inline shadow-block comments (`wavefront_kernels_taichi.py`,
  `wavefront_shadow` ~2147 area and `wavefront_shade` ~2554 area);
* `raster_shadow_trace`'s event-setup comment (`raster_taichi.py`).

### Defect 3 — shadow-free batches compile one resolve variant again

In `shade_sparse_raster_coverage` (`raster_pipeline.py`) the `else:` branch
(the `not shadow_flag` resolve) passed the live `term_mode` to
`sheet_resolve_shade`. Mode 0 has no shadow logic, so the gate cannot change
that launch's output — but it is a `ti.template()`, so forwarding the setting
compiled a second variant of the resolve per gate value for nothing. It now
passes a literal `0`, commented why; only the mode 1 / mode 2 launches take a
meaningful gate.

### The unit tests

`tests/unit_tests/test_shadow_terminator.py`, modeled closely on
`test_nested_ior.py`: module docstring stating what is pinned and why, no
`from __future__ import annotations` (probe kernels need runtime annotations),
one helper driving the `@ti.func` from a small `@ti.kernel`, sentence-named
tests, UNMARKED (outside `--fast`). One reference triangle throughout —
vertices `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, face normal +z — with data in row 1,
and every cell the query does not read filled with obviously-wrong values
(two distinct junk patterns whose normals do not agree, so even the flat
tests catch a misread). Nine invariant tests plus two settings tests:

1. flat facet → exactly zero (exact equality, four barycentric placements);
2. at each vertex → exactly zero;
3. convex patch lifts toward the normals (+z component, toward every vertex
   normal, toward every tangent plane it started below, on-or-above every
   DISPLACED vertex's plane);
4. concave patch → exactly zero (the offset only ever lifts);
5. back-facing hit lifts the other way (`delta.z < 0`), convex-negated →
   exactly zero pinned alongside;
6. degenerate normal field (zero or sub-1e-9 normal) → exactly zero;
7. magnitude bounded by the facet (`|delta| < longest edge`, non-vacuous);
8. prim past a trimmed `tri_norm` → exactly zero without reading out of
   bounds, same arrays still working for an in-prefix prim;
9. near-identical normals (~1e-8 agreement) treated as flat → exactly zero;
10. settings surface round-trip (`SETTINGS.raytracing.experimental.set`
    → `rt_settings.SHADOW_TERMINATOR`; `shadow_terminator_mode()` = 0/1/2 for
    False/True/"relax"), previous value restored in a `finally`;
11. the toggle rejected on the public `SETTINGS.raytracing` section
    (mirroring `test_nested_ior.py`).

Test 9 has teeth: replicating the pre-fix arithmetic in float32 on its exact
inputs leaves `delta ≈ (-2.5e-17, 1.9e-17, 3.1e-09)` — bitwise non-zero —
while the shipped kernel returns exact zeros.

### Two findings against the brief's test prescriptions

Both were measured against the real kernel before writing anything, and both
are properties of Hanika's estimator rather than of our implementation of it
(round 1 validated the formula end to end; neither finding changes any
rendered frame). Each test pins the brief's INVARIANT in the form the
estimator can actually honour, with the deviation documented in the test
itself.

**Finding A — the brief's tangent-plane assertion (test 3) cannot hold for
any curved convex patch.** With exactly the prescribed inputs (normals
`normalize((-0.3,-0.3,1)) / (0.3,-0.3,1) / (-0.3,0.3,1)` at the centroid),
`dot(p + delta - p_i, n_i) >= -1e-6` holds at v0 (+0.0333) but FAILS at v1
and v2 (**−0.0681329** each). The reason is structural: `delta` averages the
three clamped per-vertex lifts weighted by mutual normal cosines (< 1), so at
whichever vertex sits deepest below its tangent plane, an average of shallower,
mutually angled lifts necessarily undershoots that vertex's own plane. Equal
depths make it strictly worse (`delta·n_i = D·(n̄·n_i) < D` unless all normals
are parallel); reweighting barycentrics only moves the shortfall to another
vertex. The estimator reduces self-intersection; it is not a half-space
projection. The test asserts instead what the construction does guarantee:
`delta.z > 0`, `delta·n_i > 0` for all three i, strict improvement of every
tangent-plane gap, and on-or-above every DISPLACED vertex's plane
(`dot(p + delta - (p_i + d_i n_i), n_i) >= -1e-6`) — the last being the
definition of "origin placed on the smooth surface the normal field defines".

**Finding B — the brief's back-facing case (test 5) reads exactly zero, not
`delta.z < 0`, for the convex field.** Negating `snrm` flips the frame via
the sign rule (`sgn = -1`), and the clamp then decides what remains: for the
convex field at the centroid every mirrored depth `(p - p_i)·(-n_i)` is
positive — indeed `(p - p_i)·n_i <= 0` everywhere on this facet — so all three
clamp to zero and delta is EXACTLY `[0, 0, 0]` (measured). That is the same
"only ever lifts" principle as the concave test, seen from the other side,
and it is pinned as an explicit assertion. The direction flip itself is
demonstrated with the converging field, which DOES leave depth in the
mirrored frame: concave normals + negated `snrm` give
`delta = [-0.01694915, -0.01694915, -0.22598872]`, `delta.z < 0` — the exact
mirror image of the front-facing convex lift.

### Verification actually run (exact outputs)

1. `.venv/bin/python -m pytest -q tests/unit_tests/test_shadow_terminator.py`
   ```
   11 passed, 3 warnings in 0.71s
   ```
2. `.venv/bin/python -m pytest -q tests/unit_tests/test_environment.py tests/unit_tests/test_nested_ior.py tests/unit_tests/test_raytracing_unit.py`
   ```
   39 passed, 4 skipped, 3 warnings in 8.14s
   ```
   (The 4 skips are the pre-existing megakernel removals noted in round 1.)
3. `ruff check --no-fix tests/unit_tests/test_shadow_terminator.py benchmarks/_shadow_terminator_ab.py`
   ```
   All checks passed!
   ```
4. `ruff format --check tests/unit_tests/test_shadow_terminator.py`
   ```
   1 file already formatted
   ```
   (One earlier pass said "Would reformat"; `ruff format` was applied to the
   new test file once, then everything above was re-run clean.)
5. `.venv/bin/python benchmarks/_shadow_terminator_ab.py` — run TWICE after
   the defect fixes (once cold, once for the record):
   ```
   torus    off->on moved     (on speckle <= 50% of both ) max|d|= 41 pixels=177 (0.04%) speckle off/on/relax=41/4/38 -> OK
   sphere   acne: relax moved 24 px, 20 of them darker; on darkens 0 of the same pixels
   sphere   off->on moved     (relax darkens, on does not) max|d|= 28 pixels=8 (0.00%) speckle off/on/relax=24/21/25 -> OK
   cube     off->on identical (identical, every arm      ) max|d|=  0 pixels=0 (0.00%) speckle off/on/relax=15/15/15 -> OK
   circuit  off->on identical (identical, every arm      ) max|d|=  0 pixels=0 (0.00%) speckle off/on/relax=5/5/5 -> OK

   frames in algan_outputs/profiling
   all arms as expected
   ```

   **Defects 1 and 2 changed none of its four frames**: the headline numbers
   are byte-for-byte the ones documented pre-round-2 (torus off/on/relax =
   41/4/38 speckles; sphere acne 24 px moved, 20 darker, 0 darkened by `on`;
   cube and circuit byte-identical in every arm). The harness file itself is
   untouched.

Still unverified, unchanged from round 1: CUDA (CPU-only container),
`tests/fast` / `tests/full_renders` (outside the brief; `--fast` cannot see
this feature — no PN geometry in its scene), the Monte Carlo megakernel, and
`wavefront_shadow` at runtime (no caller compiles it).

One observation, reported rather than acted on: while this round was in
progress, three files I did not touch acquired modifications in the work tree
(mtime 02:40–02:41) — `RENDERER_WORK_QUEUE.md` (item 20's status flipped to
"built, default on", plus the "What building it found" section), 
`DESIGN_mesh_identity_open.md` §"one claim this section should not have made",
and `docs/source/advanced_user_tutorials/renderer_limitations.rst` (a new
shadow-terminator bullet). The same thing happened in round 1 with the
untracked A/B harness appearing mid-session. The content is accurate against
the shipped code and this round's defect fixes (its "flat facet's displacement
is exactly zero" claim is now true by construction), it is docs-only, and it
changes no behaviour; I left it exactly as found and did not include it in any
verification above.

## Round 3

Constraints honoured: the only files written in the tree are
`tests/unit_tests/test_wavefront_compaction.py` and this append. Nothing
under `algan/`, `benchmarks/`, `tests/full_renders/` was touched, nothing was
committed or pushed, and **no render was run** — every render-suite number
below is quoted from the diff/docs, not re-measured.

One observation, reported rather than acted on: mid-session,
`tests/fast/expected_outputs_cpu/fast.mp4` acquired a modification in the
work tree (mtime 04:17, between this round's two writes at 03:54 and 04:21).
None of this round's commands can have produced it — the only pytest run
targets `tests/unit_tests/` (no render), the probes import settings without
rendering, and the fast harness overwrites that file only under
`ALGAN_UPDATE_FAST_BASELINE=1`. This is the concurrent suite the brief said
is running in this container. I left the file exactly as found; it is not
part of this round's diff review (it is untracked-in-diff baseline content
owned by whoever is re-baselining).

### Part 1 — the regression test

Added one unmarked test,
`test_state_charge_follows_sca_width_argument_not_nested_ior_setting`
(tests/unit_tests/test_wavefront_compaction.py:108), in the file's existing
style. It monkeypatches `rt_settings.NESTED_IOR` to both `False` and `True`
and, at each value, asserts
`_wavefront_state_coefficients(SCA_WIDTH_PLAIN)["pool"] ==
_WAVEFRONT_BYTES_PER_POOL_SLOT` (the measured plain constant, referenced not
restated) and `_wavefront_state_coefficients(SCA_WIDTH_NESTED)["pool"] ==`
plain `+ (SCA_WIDTH_NESTED - SCA_WIDTH_PLAIN) * torch.float32.itemsize`. The
docstring states why it exists: a tile sized off the setting rather than off
the batch is not a wrong image, it is silently smaller tiles in every scene
that does not refract — invisible to every pixel comparison. The two
pre-existing tests pass untouched.

Exact outputs:

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_wavefront_compaction.py
...                                                                      [100%]
=============================== warnings summary ===============================
.venv/lib/python3.11/site-packages/cloup/_util.py:10
  /home/user/algan/.venv/lib/python3.11/site-packages/cloup/_util.py:10: DeprecationWarning: The '__version__' attribute is deprecated and will be removed in Click 9.1. Use feature detection or 'importlib.metadata.version("click")' instead.
    click_version_tuple = tuple(click.__version__.split('.'))

.venv/lib/python3.11/site-packages/pydub/utils.py:14
  /home/user/algan/.venv/lib/python3.11/site-packages/pydub/utils.py:14: DeprecationWarning: 'audioop' is deprecated and slated for removal in Python 3.13
    import audioop

.venv/lib/python3.11/site-packages/taichi/_lib/utils.py:70
  /home/user/algan/.venv/lib/python3.11/site-packages/taichi/_lib/utils.py:70: DeprecationWarning: 'locale.getdefaultlocale' is deprecated and slated for removal in Python 3.15. Use setlocale(getencoding()) instead.
    return path.encode(locale.getdefaultlocale()[1])

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
3 passed, 3 warnings in 0.04s
```

```
$ ruff check --no-fix tests/unit_tests/test_wavefront_compaction.py
All checks passed!
$ ruff format --check tests/unit_tests/test_wavefront_compaction.py
1 file already formatted
```

Extra verification that the test pins the defect: a scratch script in
/tmp/opencode (repo untouched) replicates the PRE-fix body — charge gated on
`rt_settings.nested_ior_mode()` — and runs this test's assertions against it:

```
assertions violated by pre-fix logic: 2
 - setting=False: nested-width pool 100 != 120
 - setting=True: plain-width pool 120 != 100
```

Both directions fail, so the test cannot pass against the old behaviour.

### Part 2 — adversarial read of the diff

Findings ranked by severity. "Nothing found" categories are listed at the end.

#### Finding 1 (medium) — "no transmissive geometry keeps the classic width" is false as stated, and `tests/fast` IS reachable by the nested-IOR default flip

The docs and this fix's rationale all gloss `ior_stack_flag`'s conjunction as
"the setting AND refraction, so a scene with no transmissive geometry keeps
the classic state width". But `refraction_flag` (tracer.py:1328-1337) has a
fourth disjunct that fires with **no transmissive material anywhere**:
`(samples <= 1 and _secondary_split_needed(merged, analytic_raster))`.
`_secondary_split_needed` (tracer.py:246-288) returns True for any batch with
a REFLECTIVE primitive when analytic AA is active — and by
scene_builder.py:1561-1564 *every PBR triangle is reflective*
(`refl > 0 or (refl >= 0 and ior > 1+1e-4)`; `MeshStandardMaterial` always
packs per-corner `refractive_index = 1.5` even with `ior=None`,
primitives.py:516-520, and packs `reflectivity = metalness`, :513).
`analytic_raster` defaults on (`analytic_raster_route_active`,
tracer.py:1101; `ANALYTIC_AA`/`ANALYTIC_AA_TRI` default True,
settings.py:1129/1152).

So under the flipped default, for ANY scene containing one PBR triangle or
one mirror: `refraction_flag = 1` → `ior_stack_flag = 1` (tracer.py:1348-1350)
→ rs_sca widens 7→12 columns, the stack kernel variants compile, and auto
tile sizing charges +20 bytes/slot via the very `state_sca_width` argument
this fix threaded through.

Concretely: **tests/fast's scene contains
`MeshStandardMaterial(color=RED, roughness=0.35, metalness=0.4)`
(tests/fast/scene.py:116-118)**, so its batch takes the wider state and new
kernel variant after the flip. That contradicts the claim that "`tests/fast`'s
scene cannot be affected by either change." What is true: shadows are absent
from that scene (its docstring says so deliberately; the harness sets no
shadow settings), so the terminator change cannot reach it, and the stack is
runtime-inert without transmission (no transmitted child is ever spawned, so
nothing pushes/pops media; tiles partition pixels), so the rendered bytes most
likely stay identical — but that is an empirical argument about inertness, not
the "untouched, gate or no gate" by-construction claim the docs make.

Wrong claims, each saying the predicate is transmissive geometry:
* settings.py NESTED_IOR block ("only in a batch that carries refraction ...
  so a scene with no transmissive geometry keeps the classic state width and
  the classic kernels");
* RENDERER_WORK_QUEUE.md item 5 ("a scene with no transmissive geometry keeps
  the classic `rs_sca` width and the classic kernel variants and is
  byte-identical"; likewise "Only a batch that actually refracts" — ambiguous
  at best);
* DESIGN_mesh_identity_open.md §H ("a scene with no transmissive geometry is
  untouched, gate or no gate");
* tracer.py:604-608, `_wavefront_state_coefficients`' own docstring.

Item 5's narrower sentence "Of the pixel suites, only `materials_and_lighting`
carries transmission at all" is TRUE as written (grepped all six full_renders
scenes + fast: `transmission=`/`ior=` appear only there) — but transmission is
not what gates the width, so it does not support the surrounding claim.
Operational consequence to expect: after the default flip, `solids_and_camera`
and `tests/fast` compile different kernel variants (cold compiles move between
runs); if either suite's baseline ever moves, this conjunction is where to
look first.

Trigger: already armed — the default is flipped in this same diff. Confidence:
high on the mechanism (every link read in code); medium-high that pixels do
not move (not renderable here).

#### Finding 2 (low-medium) — `set_shadow_terminator` coerces unexpected inputs into surprising arms (measured)

Probed every input the brief named, host-side only:

```
set_shadow_terminator(              True) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(             False) -> SHADOW_TERMINATOR=False    mode=0
set_shadow_terminator(           "relax") -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(           "RELAX") -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(         " relax ") -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(                 2) -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(                 1) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(                 0) -> SHADOW_TERMINATOR=False    mode=0
set_shadow_terminator(               2.5) -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(              None) -> SHADOW_TERMINATOR=False    mode=0
set_shadow_terminator(   np.float32(2.0)) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(   np.float64(2.0)) -> SHADOW_TERMINATOR=2        mode=2
set_shadow_terminator(       np.int32(2)) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(   np.bool_(True)) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(                 3) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(                -1) -> SHADOW_TERMINATOR=True     mode=1
set_shadow_terminator(                "") -> SHADOW_TERMINATOR=False    mode=0
```

Which are wrong:
* `2.5` → **mode 2**, the diagnostic relax-only arm, via `int(enabled) == 2`
  truncation. A fractional scalar is truthy numeric input; silently selecting
  the arm whose images are documented as knowingly wrong is the worst of the
  three readings. Should be plain-on (or raise).
* numpy scalars route inconsistently: `np.float64(2.0)` → mode 2 (it
  subclasses Python float) while `np.int32(2)` and `np.float32(2.0)` → mode 1
  (they fail both isinstance branches and fall through to `bool`). The "same"
  number selects different arms depending on dtype.
* `None` → mode 0 silently; undocumented (docstring promises only
  True/False/"relax"/2). Benign but should be stated or rejected.
* `3`, `-1`, `""` → truthiness/falsiness; defensible.

Trigger: `SETTINGS.raytracing.experimental.set(shadow_terminator=<any of the
above>)` from user code — this is a public surface
(raytracing_settings.py `_SETTER_OVERRIDES`). Confidence: high (measured).

#### Finding 3 (low) — "+4 f32 per ray" understates the widening by one column

RENDERER_WORK_QUEUE.md item 5 ("A scene that does refract pays 4 extra f32
per ray of state") and the settings.py NESTED_IOR block ("widens ``rs_sca``
by 4 f32 per ray") are arithmetically wrong against the code's own constants:
`SCA_WIDTH_NESTED − SCA_WIDTH_PLAIN = 12 − 7 = 5` f32 columns per ray/slot
(the depth counter column plus four stack entries) = +20 bytes. The round-3
brief's own "(+4 f32 per pool slot)" repeats the same slip. My regression test
is immune — it asserts the `(SCA_WIDTH_NESTED - SCA_WIDTH_PLAIN) *
torch.float32.itemsize` formula, exactly as instructed. Confidence: high.

#### Finding 4 (low) — the flat-mesh byte-identity mechanism is misdescribed for Algan's own flat family

renderer_limitations.rst ("On a flat-shaded mesh every vertex normal equals
its face normal, the displacement is exactly zero") and item 20 ("All three
vertex normals equal ⇒ every `d_i` is zero") describe the constant-field
equality test as what zeroes delta on flat meshes. For Algan's built-in flat
solids the packed corner normals are not equal-to-the-face-normal — they are
ZERO: `Polyhedron` builds faces as `TriangleTriangulated(corners, ...)`
passing no normals (shapes_3d.py:1293-1301), `TriangleTriangulated` passes
`normals=None` through, and `get_render_primitives` substitutes
`torch.zeros_like(locations)` (shapes_2d.py:505-510). Delta is exactly zero
there because of the degenerate-normal guard (`n.norm() > 1e-9` fails for all
three, shading_taichi.py) — which is equally airtight, so the CONCLUSION
(byte-identity for Cube/Polyhedron scenes) stands. The equality-test wording
does apply to imported/authored meshes carrying duplicated face normals.
Confidence: high.

#### Nothing found in the remaining categories

1. **Kernel parameters vs call sites — nothing found.** All four modified
   signatures walked against every launch, argument by argument:
   `sheet_resolve_shade` ×3 (raster_pipeline.py:2059/2170/2193 — including the
   mode-0 dummy launch, whose ten literals map sheet_accept←dummy_i,
   pos/snrm/fnrm←dummy_f3×3, frame/msk←dummy_i×2, dp←dummy_f6, toff←dummy_f3,
   event_id←dummy_i, vis←dummy_vis, with pre_args ending at
   `env_in_composite` and post_args beginning at `covered_idx`);
   `raster_shadow_trace` ×1 (:2115, `event_dp, event_toff` adjacent exactly as
   in the signature, `term_mode` last); `wavefront_shade` ×2 (tracer.py:2492,
   2932 — inserted slot sits between `deferred_shadows`(0) and
   `skip_unlit_normal` in both); `wavefront_shadow` — no caller exists (the
   docs say so too), and its inline edit mirrors `wavefront_shade` correctly.
   `_shadow_terminator_delta`'s three call sites all pass barycentrics in the
   `(w0, a, b)` convention with `w0 = 1-a-b`, oriented `snrm`, and matched
   tri_pos/tri_norm pairs.
2. **`event_toff` lifetimes and sizes — nothing found.** Producer writes rows
   under `sheet_accept ∧ ti.static(shadow_term == 1)` inside the triangle-only
   `if not fetched_bez:` block — the identical condition that sets
   `sheet_accept` — so EVERY accepted row is written (zero-delta rows included;
   bezier sheets are never accepted). Allocation is full-size iff
   `term_mode == 1`; the consumer reads iff `ti.static(shadow_term == 1)`;
   `num_events == 0` skips the trace entirely (`if num_events:`); the
   `sec_aa > 1` pairing around `event_dp` is replicated exactly for toff
   (`index_select` by the same `acc_idx`). No read of an unwritten row exists
   in any mode combination I could construct (term_mode ∈ {0,1,2} × sec_aa ∈
   {1,>1} × num_events ∈ {0,>0}).
3. **`ti.static` gates — nothing found.** All gates branch on template
   constants (`mode`, `shadow_term`, `sec_aa`, `shadows`, `deferred_shadows`);
   `lifted = 0` / `delta` are initialised before/outside the gates in all
   three trace-side kernels, so the `shadow_term == 0` compilation leaves no
   uninitialised reads; the wavefront fan block containing the new code sits
   inside `ti.static(shadows != 0)` (wavefront_kernels_taichi.py:2510), so
   shadowless scenes compile none of it; passing literal `0` at the mode-0
   launch to avoid a second compiled variant is deliberate and correct.
4. **Tri-state mode 2 — nothing found structurally.** Traced through
   allocation ((1,3) dummy), build (write gated `== 1`, compiles out),
   trace (reads gated `== 1`, never executed; `lifted = 1` everywhere else is
   the documented diagnostic), shading (mode-2 pass consumes only
   `shadow_vis`, whose rows are host-initialised 1.0 and keyed by
   `sheet_event_id >= 0`). Reachable only via env var or explicit setter.

#### Claims checked and found accurate

Shadow-ray lift `10 * MIN_HIT_DISTANCE` = 1e-3 world units and stop-short
`20 * MIN_HIT_DISTANCE` = 2e-3 (MIN_HIT_DISTANCE = 1e-4,
raytrace_kernels_taichi.py:98) — matches renderer_limitations.rst verbatim.
"`fnrm.dot(wis) > 1e-3` rejects within ~0.06 degrees" — arcsin(1e-3) ≈
0.057°, correct. `IOR_STACK_DEPTH = 4` ↔ "up to four nested media / a fifth
loses track", correct. "wavefront_shadow carries the change but has no
caller" — honest, confirmed by grep. The trimmed-`tri_norm` guard semantics
(`prim < shape[1]`, order-preserving prefix) match
`_flat_triangle_normal_trim`'s existing convention, and the mem-trim pairs
(`tri_pos_t`/`tri_norm_t` through the same `perm`) keep vertex positions and
normals consistent wherever delta runs. benchmarks/_shadow_terminator_ab.py
measures exactly what items 20/§H quote from it (speckle definition, three
arms, cube/circuit must-be-identical assertions); its recorded numbers are
quoted, not re-measured. tests/fast's scene deliberately excludes shadows
(scene.py:44-46) and the harness enables none — the terminator half of
"tests/fast cannot be affected" holds; see Finding 1 for the nested-IOR half.
