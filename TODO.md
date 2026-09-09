# Algan TODO — prioritized remaining work

Reviewed against `master` commit `f10cc230a108863d02980fc27079254473ae7de3`
on 2026-09-09. This is a source-verified backlog, not a list of every possible
feature. Unmerged branches are not counted as implemented. Update or remove an
entry when its acceptance criteria land; keep historical measurements in the
relevant design or benchmark report.

The ordering favors visible correctness and image quality, then validation and
measured performance. It is an engineering priority judgment, not a benchmark
prediction. [RENDERER_WORK_QUEUE.md](RENDERER_WORK_QUEUE.md) maps the older renderer
item numbers to their current status.

## 1. Filter minified textures on primary and secondary hits

**Current gap.** The UV texture samplers in
[`wavefront_kernels_taichi.py`](algan/rendering/raytracing/wavefront_kernels_taichi.py)
and [`shading_taichi.py`](algan/rendering/raytracing/shading_taichi.py) use bilinear
sampling without a mip level or a texture footprint. Geometric analytic AA does
not filter detail *inside* a textured surface. Small or oblique textures can
alias in both the deterministic renderer and the path tracer.

**Work.** Add a compact mip representation and inexpensive footprint/LOD selection
for raster primaries and reflected/refracted paths. A sheet's screen area alone
is insufficient: the footprint must account for the UV mapping and ray spread.
Keep color-space, opacity, normal-map and scalar-property filtering semantics
explicit. The glossy-reflection pyramid in
[`DESIGN_glossy_prefilter.md`](algan/rendering/raytracing/DESIGN_glossy_prefilter.md)
is a useful implementation precedent, not a UV-texture pyramid to reuse blindly.

**Done when.** Checkerboards, thin alpha features and material/normal maps remain
stable under minification, grazing views, camera motion and reflections on both
renderers. Include odd texture sizes, wrap seams and animated texture windows;
measure build cost, extra memory and warm render cost on representative hardware.

## 2. Compensate very rough dielectric energy loss

**Current gap.** `_pt_glass_f_pdf` and `_pt_sample_glass` in
[`path_tracer_taichi.py`](algan/rendering/raytracing/path_tracer_taichi.py) implement
a unified single-scatter rough dielectric. Opaque GGX has multiple-scattering
compensation, but rough glass still omits repeated microfacet scattering.

**Work.** Couple reflection and transmission compensation rather than applying
the opaque reflection-only correction to both. Preserve matching evaluation,
sampling and PDFs, total internal reflection, nested relative IORs and the
radiance eta-squared convention. See the glass implementation record in
[`DESIGN_path_tracer_roadmap.md`](algan/rendering/raytracing/DESIGN_path_tracer_roadmap.md).

**Done when.** White-furnace and roughness/IOR sweeps demonstrate the intended
energy behavior, with reciprocal-interface and TIR cases and no smooth-glass
regression. Document any approximation rather than calling it exact transport.

## 3. Finish physical area-emission semantics across the API

**Current gap.** Rectangular emitter geometry is already merged into the path
tracer's normal scene build and is visible to camera and continuation rays.
However, [`lights.py`](algan/rendering/lights.py) still defaults to the legacy
`decay=0, distance=0` convention. Distance-independent physical emitter radiance
corresponds to `decay=2, distance=0`; other settings retain artistic falloff.

**Work.** Define the public intensity/radiance units and the migration of legacy
falloff explicitly. Apply the same convention to direct-light sampling, camera
hits and BSDF-sampled emitter hits; do not silently change just one estimator.
The current implementation is documented in
[`DESIGN_physical_area_lights.md`](algan/rendering/raytracing/DESIGN_physical_area_lights.md).

**Done when.** Equivalent emitter encounters agree at multiple distances, sizes,
orientations and bounces; MIS does not double-count emission; the public examples
and compatibility policy explain any deliberately changed output.

## 4. Improve antialiasing where surfaces cross inside a pixel

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

## 5. Define closed-solid opacity consistently for deterministic continuations

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

## 6. Make renderer-audit inputs equivalent before drawing new conclusions

**Current gap.** The comparison bridges in
[`algan_render.py`](benchmarks/renderer_audit/algan_render.py) and
[`three_render.mjs`](benchmarks/renderer_audit/three_render.mjs) disagree when a
material omits its type or color. The Algan bridge ignores the JSON camera's
`up`, `near` and `far`, and sphere tessellation is not shared. The corrected
[`SPEC.md`](benchmarks/renderer_audit/SPEC.md) documents these limitations instead
of promising identical scenes.

**Work.** Define shared defaults, validate the input format, map supported camera
fields, and distinguish intentional geometry/material differences from missing
translation. Test the bridge outputs before using new renders to rank defects.
Also repair or retire `_prespawn_invisibility_check.py`, which imports the removed
`memory_utils.empty_cache` name; do not treat old probe commands as acceptance
coverage merely because their files still exist.

**Done when.** Schema/default tests and camera-construction checks agree across
the bridges, deliberately non-equivalent panels are labeled, and referenced
acceptance harnesses import and run against the current API.

## 7. Make release and documentation validation reproducible

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

## 8. Profile and reduce avoidable CPU memory reclamation

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

## 9. Remove confirmed legacy render experiments in a separate code change

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

Planar circuits are intentionally unlit; use `TriangulatedBezierCircuit` when a
vector shape needs surface lighting. A native lit-circuit path is a product
choice, not a broken material that merely needs enabling. TLAS/BLAS instancing
and further sort/resolve rewrites should be justified by fresh scene-specific
profiles rather than promoted solely because an old plan proposed them.
