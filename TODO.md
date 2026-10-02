# Algan TODO — prioritized remaining work

Reviewed against `master` commit `f10cc230a108863d02980fc27079254473ae7de3`
on 2026-09-09. This is a source-verified backlog, not a list of every possible
feature. Unmerged branches are not counted as implemented. Update or remove an
entry when its acceptance criteria land; keep historical measurements in the
relevant design or benchmark report.

The review's item 1, filtering minified textures on primary and secondary hits,
landed in `7bcc19c` (default-on UV mip anti-aliasing,
[`DESIGN_texture_antialiasing.md`](algan/rendering/raytracing/DESIGN_texture_antialiasing.md))
and was removed on 2026-09-30; the items after it each moved up one number.
Its deliberate limits are listed under later design work below.

The ordering favors visible correctness and image quality, then validation and
measured performance. It is an engineering priority judgment, not a benchmark
prediction. [RENDERER_WORK_QUEUE.md](RENDERER_WORK_QUEUE.md) maps the older renderer
item numbers to their current status.

## 1. Improve antialiasing where surfaces cross inside a pixel

**Current gap.** Sheet compaction already computes per-sample depth information.
That repairs which surface wins samples, but it is not an exact area blend of an
interpenetration seam. A scalar/centroid depth and discrete ownership cannot
represent every within-pixel depth crossing.

**Work.** Evaluate a depth-plane or equivalent inexpensive crossing representation
in [`sheets.py`](algan/rendering/raytracing/sheets.py),
[`sheet_compact_taichi.py`](algan/rendering/raytracing/sheet_compact_taichi.py) and
[`sheet_resolve_taichi.py`](algan/rendering/raytracing/sheet_resolve_taichi.py).
Preserve coplanar layer order, transparent stacking and same-surface identity.

**Done when.** Crossing opaque and transparent meshes converge toward a
supersampled reference without breaking silhouette/tiling fixtures. Reproduce
video defects with multi-frame renders as well as stills: frame-window slicing
has previously changed identity and hidden failures from still-only probes.

## 2. Define closed-solid opacity consistently for deterministic continuations

**Current gap.** Primary sheet compositing and the path tracer have closed-shell
accounting, but the deterministic wavefront's treatment of a solid encountered
through a reflection does not have the same general one-shell opacity contract.
[`agent_guidance/rendering.md`](agent_guidance/rendering.md) records the boundary;
[`DESIGN_mesh_identity_open.md`](algan/rendering/raytracing/DESIGN_mesh_identity_open.md)
retains the design discussion.

**Work.** Specify how front/back encounters, re-entry, overlapping solids and
transmission exemptions interact before sharing or extending the accounting.
Do not equate artistic shell opacity with Beer–Lambert absorption.

**Done when.** The same closed solid has the specified opacity in direct,
reflected and nested views, with explicit controls for thin/open surfaces and
physical glass. Validate continuation retries and surface-limit reporting too.

## 3. Make renderer-audit inputs equivalent before drawing new conclusions

**Completed in the October bug-audit fixes.** The comparison bridges in
[`algan_render.py`](benchmarks/renderer_audit/algan_render.py) and
[`three_render.mjs`](benchmarks/renderer_audit/three_render.mjs) now share render,
camera and material defaults. Both apply camera `up`, `near` and `far`, and reject
unknown camera keys. Python/JavaScript normalization and camera construction have
regression coverage. [`SPEC.md`](benchmarks/renderer_audit/SPEC.md) records the
remaining differences in tessellation, far clipping and material semantics.

**Remaining work.** Broaden schema validation for geometry and material fields.
Repair or retire `_prespawn_invisibility_check.py`, which imports the removed
`memory_utils.empty_cache` name; do not treat old probe commands as acceptance
coverage merely because their files still exist.

**Done when.** Schema/default tests and camera-construction checks agree across
the bridges, deliberately non-equivalent panels are labeled, and referenced
acceptance harnesses import and run against the current API.

## 4. Make release and documentation validation reproducible

**Current gap.** Structural Sphinx checks, directive checks and pixel suites exist,
but historical reports are not proof that the next release candidate passes.
Missing baseline assets now fail comparisons, but CI exclusions and the explicit
macOS unbaselined opt-out can still leave pixels untested. The current
Markdown audit also found drift that a Sphinx-only build cannot see.

**Work.** Check the exact release SHA, installed wheel contents, matching baseline
manifest/assets and CPU/CUDA/MPS coverage. Add automated repository-Markdown
link/anchor checks and source-backed checks for documented settings and examples.
Keep missing baseline assets and expected backend failures explicit.
The MPS manifest still contains `ti.real_func` early return; isolate that compiler
defect and remove its strict xfail only after a hardware-confirmed fix. Verify
rendered documentation examples separately from a structural build.

**Done when.** A release's recorded evidence names its source SHA, compiler build,
backend, baseline keys and skips; the published docs match that release; broken
local documentation links and obsolete public examples fail validation. See
[`RELEASE_RUNBOOK.md`](RELEASE_RUNBOOK.md) and [`tests/README.md`](tests/README.md).
Do not regenerate baselines merely to hide an environment/toolchain mismatch.

## 5. Profile and reduce avoidable CPU memory reclamation

**Current gap.** [`_gpu_memory_pressure`](algan/utils/memory_utils.py) falls back to
`True` without GPU telemetry, so `release_torch_memory(force_gc=False)` can still
collect on every CPU-only call. Host/cgroup telemetry and native-memory recovery
already exist; MPS also has its own pressure check. The old claim that *all*
non-CUDA devices always take the same path is obsolete.

**Work.** Measure collection and allocator costs on CPU-only renders, then make
routine reclamation depend on meaningful pressure without weakening forced OOM
recovery, cgroup protection or long-running scene cleanup. Inspect device
selection as well as device availability when evaluating the predicate.

**Done when.** Warm end-to-end comparisons show a benefit, steady-state collections
are bounded, cyclic garbage is still reclaimed, and low-memory/retry tests pass.
Do not treat the older profiling percentages as the current bottleneck ranking.

## 6. Remove confirmed legacy render experiments in a separate code change

**Current gap.** [`bloom.py`](algan/rendering/post_processing/bloom.py) still contains
unused `bloom_filter_old`/`bloom_filter_conv` experiments and a compatibility probe
for a removed glow helper. An SMAA module also exists without a normal
post-processing route. Their presence is not evidence of supported public features.

**Work.** Audit callers and imports, remove unused implementations and stale guards,
or deliberately integrate and test an alternative that is still wanted. Keep
this behavior-affecting cleanup separate from documentation-only changes.

**Done when.** No public export, import or supported post-processing configuration
depends on the removed code, and bloom/alpha/export regression tests remain green.

## Later design work, not missing implementations

Homogeneous scattering volumes and bounded random-walk subsurface scattering,
light-tree selection, adaptive sampling, environment-map sampling and OIDN-based
denoising are already implemented. Do not reopen them as absent features.
Heterogeneous density/ratio tracking, stronger difficult-caustic sampling and
more capable temporal denoising are extensions with separate cost and quality
criteria, described in the path-tracer roadmap.

Custom fragment scatter currently preserves the parent medium rather than
declaring a nested-medium transition. Extending that injection signature needs
a separate contract and tests; it is not the already-completed built-in nested
IOR work.

UV texture minification filtering is implemented and on by default
(`SETTINGS.raytracing.texture_antialiasing`). Its footprint is a deliberately
cheap isotropic pixel cone, grown along the accumulated camera-path length.
Anisotropic filtering for grazing views (which the isotropic filter overblurs),
ray differentials that track curved-mirror magnification and refractive
focusing, and prefiltered environment-map lookups (environment maps are still
sampled at full resolution) are extensions with their own cost and quality
criteria, not a missing filter. The repository records no GPU measurement of
the mip chain's build, memory or warm-render cost (the design defers to its
PR's validation notes), so measure before tuning it.

Planar circuits are intentionally unlit; use `TriangulatedBezierCircuit` when a
vector shape needs surface lighting. A native lit-circuit path is a product
choice, not a broken material that merely needs enabling. TLAS/BLAS instancing
and further sort/resolve rewrites should be justified by fresh scene-specific
profiles rather than promoted solely because an old plan proposed them.
