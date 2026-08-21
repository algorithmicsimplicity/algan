# OX_REPORT.md — §I self-shadow rejection by identity (work-queue item 3)

Implementation of `DESIGN_mesh_identity_open.md` §I as specified by item 3 of
`RENDERER_WORK_QUEUE.md`. Nothing is committed; everything is in the working
tree (7 modified files, no new files in the repo).

## What changed, file by file

### `algan/rendering/raytracing/settings.py`
New experimental toggle, following the exact pattern of its neighbours
(module global + env-var default + setter), placed directly after
`set_shadow_anyhit`:

- `SHADOW_IDENTITY_REJECT = env_flag("ALGAN_SHADOW_IDENTITY_REJECT", False)` — **default OFF**.
- `set_shadow_identity_reject(enabled)` setter.
- A comment block stating the acceptance test, the "same mesh AND near-zero t,
  never same mesh" rule, and what keeps today's epsilon.

### `algan/environment.py`
Declared `"ALGAN_SHADOW_IDENTITY_REJECT"` in `_IMPORT_TIME_VARIABLES`
(alphabetical, between `ALGAN_SHADOW_ANYHIT` and `ALGAN_SHEET_MASK_KERNEL`).
It belongs there because the `env_flag` call sits at module level in
`settings.py`, exactly like every sibling toggle;
`tests/unit_tests/test_environment.py` enforces this classification and passes.

### `algan/rendering/raytracing/raytrace_kernels_taichi.py`
- New `@ti.func _shadow_identity_t_min(f, prim, src_sid, tri_obj, ident)`:
  returns the acceptance floor for one candidate triangle hit. Default
  `MIN_HIT_DISTANCE`; under the compile-time gate (`ti.static(ident != 0)`)
  and a per-ray runtime id (`src_sid >= 0`), a hit on a *different* mesh gets
  floor `0.0` and a hit on the ray's own mesh keeps `MIN_HIT_DISTANCE`. The
  rejection is per hit ("same mesh AND near-zero t"), never "same mesh" — a
  concave solid still shadows itself.
- Threaded `(src_sid, tri_obj: ti.template(), ident: ti.template())` through
  the shared chain: `_nearest_triangle_hit`, `_nearest_surface_g`,
  `_nearest_surface` (wrapper — passes sentinel `-1, tri_pos, 0`),
  `_collect_hits`, `_anyhit_opaque_tri`, `_shadow_anyhit_opaque`,
  `_shadow_occluded`, `_shadow_march_occluded`, `_shadow_gather_occluded`.
- The acceptance test itself changed in exactly three places:
  `_nearest_triangle_hit`, `_collect_hits`'s triangle arm, and
  `_anyhit_opaque_tri`. With `ident == 0` the helper compiles down to
  `MIN_HIT_DISTANCE`, so every predicate reduces to precisely today's
  expression — that is the byte-identity argument for the default path.

### `algan/rendering/raytracing/wavefront_kernels_taichi.py`
All 8 call sites of the shared funcs updated with sentinels so the build does
not break: 4× `_nearest_surface_g` (both traverse kernels, opaque-closest and
opaque-prepass arms), 2× `_collect_hits`, 2× `_shadow_occluded` (the legacy
deferred stage and `wavefront_shade`'s inline block). All pass
`(-1, tri_pos, 0)`: the megakernel has no source identity available (and
per §H's note, `wavefront_shade` must not gain another ndarray). Identity
rejection therefore engages only on the sheet route's shadow queue — which is
where the design says the source id exists.

### `algan/rendering/raytracing/raster_taichi.py`
`raster_shadow_trace` gained two appended parameters: `tri_obj
(ti.types.ndarray())` and `shadow_identity (ti.template())`. When enabled it
unpacks the source id from `event_msk` bits 16+ (`((msk >> 16) & 0xFFFF) - 1`,
0 = none → `-1`) and masks `pid_e` to its low byte; otherwise it leaves both
reads exactly as they were. Forwards `(src_sid, tri_obj, shadow_identity)` to
`_shadow_occluded`. Docstring updated.

### `algan/rendering/raytracing/raster_pipeline.py`
- New host helper `_pack_shadow_source_ids(merged, ev_msk, ev_frame, ev_ref)`:
  packs `sid + 1` into bits 16–31 of each accepted event's mask word, using
  the same row mapping the kernel uses (`event_frame % tri_obj.shape[0]`).
  Per-event guards fall back to the sentinel (classic epsilon) rather than
  misrejecting: non-triangle ref, `sid` outside `[0, 0xFFFE]`, or an existing
  material pipeline id ≥ 256 that would collide with bit 16.
- `shade_sparse_raster_coverage`: when `rt_settings.SHADOW_IDENTITY_REJECT`
  is on, packs the accepted events' masks after compaction and passes
  `merged["tri_obj"]` plus the flag to `raster_shadow_trace`; when off it
  passes a 1-element dummy and `0`.

### `algan/settings/raytracing_settings.py`
Added `"SHADOW_IDENTITY_REJECT"` to `_FIELD_TO_LEGACY` so the toggle is also
reachable as `SETTINGS.raytracing.experimental.shadow_identity_reject`
(reads worked regardless via the legacy fallback; this makes writes legal too,
so it does not join work-queue item 16's unreachable list).

## Toggle name and environment variable

- Setting: `SHADOW_IDENTITY_REJECT` in
  `algan.rendering.raytracing.settings`, exposed at
  `SETTINGS.raytracing.experimental.shadow_identity_reject`.
- Environment variable: `ALGAN_SHADOW_IDENTITY_REJECT` (import-time; set
  before `import algan`, or flip per batch via the experimental view).
- Default: **OFF**. With it off the renderer behaves exactly as today.

## Test results actually observed

1. Cheapest import check:

   ```
   $ .venv/bin/python -c "import algan; ..."
   import ok
   toggle: False
   ```

2. Full fast suite (after a cold recompile of the edited kernels):

   ```
   $ .venv/bin/python -m pytest -q --fast
   FAILED tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline
   AssertionError: fast.mp4 differs from its baseline by up to 40 channel
   values (worst at frame 6); see /home/user/algan/tests/fast/output_errors/fast.mp4
   assert 40 <= 2
   fast suite: 17s of its 75s budget (23%)
   1 failed, 271 passed, 1240 deselected, 3 warnings in 17.19s
   ```

   That is the required pre-existing failure, unchanged: **exactly 40 channel
   values, worst at frame 6**. Everything else (271 tests) passed.

3. Environment registry test:

   ```
   $ .venv/bin/python -m pytest -q tests/unit_tests/test_environment.py
   19 passed, 3 warnings in 8.19s
   ```

4. Lint: `ruff check --no-fix algan/` reports 15 findings; I verified by
   stashing my changes that all 15 are pre-existing on the clean tree (same
   rules; only line numbers shifted in files I touched). None introduced.
   `ruff format --check` passes on the four non-taichi files I touched (the
   `*_taichi.py` files are excluded from formatting by config).

## The A/B smoke test, and what its max-diff-of-0 means

I wrote `/tmp/opencode/shadow_identity_smoke.py` (outside the repo): a slab
plus a small block under a deliberately grazing directional light and a point
light, `shadows=True`, rendered at `SMOKE_TEST` quality in two processes,
`ALGAN_SHADOW_IDENTITY_REJECT=0` vs `=1`. Both rendered end to end without
error — so the ON path (host-side packing, the new `raster_shadow_trace`
kernel variant, the identity-aware traversal) executes. Decoding and diffing:

```
shapes: (2, 32, 32, 3) (2, 32, 32, 3)
frames compared: 2
max channel diff: 0
per-frame max: [0, 0]
pixels moved (|d|>2): 0
```

**Blunt reading of the zero:** this result is expected and proves much less
than it looks like it might.

1. My scene geometry was wrong for the purpose. The block ended up floating
   ~0.47 world units above the slab (I misplaced it; the first script version
   also crashed on `Mob.scale`'s signature before I got a render at all).
   There was no contact anywhere in the scene, so the regime the feature
   targets did not exist in it.
2. Even with correct contact, the affected band is far below one pixel at
   this scale and resolution. A cross-mesh blocker only changes the outcome
   when its hit lands in `t ∈ (0, 1e-4]` along the shadow ray; with my ~8°-
   elevation light that is a world-space band roughly `1e-4/sin(8°) ≈ 7e-4`
   units wide around the contact. At 32×32 over a ~2-unit frame of view, one
   pixel spans ~0.06 units — about 100× the entire affected band. A zero diff
   here says nothing about whether the mechanism works.
3. What the smoke test genuinely establishes: the toggle-ON path runs without
   crashing, and toggle-OFF output is unchanged by the presence of the new
   code (also established by the full `--fast` suite above).

**What I did NOT establish**, because I stopped when asked to write this
report: that the feature actually *engages* — i.e. that accepted events
really carry packed ids and the kernel really accepts cross-mesh hits below
`MIN_HIT_DISTANCE`. My next step, not taken, was to demonstrate engagement
directly: either shrink the whole scene ~100× (so the erased band becomes a
large fraction of the object's shadow, which is exactly the small-scale regime
§I exists for) and render both arms at LD/MD, or instrument
`_pack_shadow_source_ids`/the trace launch to count packed-vs-sentinel
events. Neither was run. **The feature should be treated as plumbed but
pixel-unverified.**

## Other things I could not do, or am unsure about

- **No CUDA.** This container is CPU-only. Everything above is the CPU path;
  the new kernel variants have never been compiled or run on CUDA.
- **Only the default shadow mode was exercised.** The threading covers all
  three any-hit modes (2 deferred, 3 opaque-only, 4 gather — including
  `_collect_hits`, which the design's five-function list omitted but the
  gather march needs), but I only ever rendered with the default mode.
  Same for the refit vs classic BVH walks and the `sec_aa > 1` sub-pixel
  variant: compiled per variant, never run.
- **No unit test added** for `_shadow_identity_t_min` or the packing helper.
  The task's deliverable was the change plus verification via the existing
  suites; a dedicated test (e.g. a two-mesh contact scene asserting the
  ON/OFF pixel difference is non-zero in the contact band) is the obvious
  follow-up and is what would settle the engagement question above.
- **Two deliberate deviations from the design text**, both noted in code
  comments: (a) the design says `event_msk` uses "only its low 4 bits",
  but bits 8+ already carry the material pipeline id, so the source id is
  packed at bits 16–31 and pipeline ids are bounded to 8 bits by a host-side
  guard (events that do not fit keep the classic epsilon); (b) the trace
  kernel masks `pid_e` to its byte only when the feature is on, so the off
  path keeps today's unmasked read even for hypothetical ≥256 pipeline ids.
- Bezier circuits keep `MIN_HIT_DISTANCE` as blockers (they have no
  per-triangle identity); the design's function list implies the same.
