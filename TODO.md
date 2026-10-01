# Algan TODO — prioritized remaining work

Re-reviewed against `master` commit `7e2ffc02173a1f4120c40e51eb0f65d5d4f79cc2`
on 2026-10-01 (first reviewed at `f10cc230`, 2026-09-09). This is a
source-verified backlog, not a list of every possible feature. Unmerged
branches are not counted as implemented. Update or remove an entry when its
acceptance criteria land; keep historical measurements in the relevant design
or benchmark report.

Item 2 below landed on 2026-09-30 and item 5's stated gap was fixed. Both are
kept, marked, rather than removed, so the item numbers that
[RENDERER_WORK_QUEUE.md](RENDERER_WORK_QUEUE.md) cites stay valid.

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

## 2. Define closed-solid opacity consistently for deterministic continuations — landed

**Landed in `7314aaf`** (merged 2026-09-30). The deterministic wavefront now
pairs closed-shell crossings on every straight segment, reflections and
classic primary rays included, so a closed solid at `opacity < 1` composites
once in direct and reflected views on both deterministic front ends.
[`shell_alpha.py`](algan/rendering/raytracing/shell_alpha.py) assigns dense
shell IDs held in a per-ray bitset in `rs_sca`, with no fixed nesting cap;
physical transmission is excluded at packing, so glass still evaluates both
interfaces; `ALGAN_SOLID_SHELL_ALPHA=0` is the control. Tests in
`tests/unit_tests/test_hybrid_transport.py` cover mirror opacity, re-entry,
more than four overlapping shells, state across event batches and the tile
planner's accounting. [`agent_guidance/rendering.md`](agent_guidance/rendering.md)
documents the contract.

**Residual.** The memory-trim permutation is disabled for batches that carry
shell IDs until its shell-ID mapping is implemented. That is a memory and
efficiency gap, not an opacity one.

## 3. Make renderer-audit inputs equivalent before drawing new conclusions

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

**Fixed since the first review.** `b284532` made
[`_gpu_memory_pressure`](algan/utils/memory_utils.py) answer `False` for a CPU
render (it answered `True` without GPU telemetry, so every steady-state
`release_torch_memory(force_gc=False)` on a CPU render paid a full collection);
`test_an_unpressured_cpu_reclaim_skips_gc` pins it. `52df5e9` added a back-off
for host-memory reclaims that keep freeing nothing (Windows). And a finished
render no longer leaves its arena in cyclic garbage: `render_batch_raytraced`'s
self-recursive `render_chunk` closure held the job's `ManualMemory`, which only a
full collection freed (2.7 GB per render on the explainer benchmark, PREVIEW,
CPU); it now clears that cycle itself
(`test_a_finished_render_leaves_no_closure_cycle_holding_its_arena`).

**Current gap.** No warm end-to-end comparison of the CPU change is recorded.
The one full `gc.collect()` before each render job (`scene_excluded_from_gc`)
remains and costs 0.13–0.39 s on that benchmark. It is load-bearing: the job
sizes a fresh arena from the memory free just after it, so cyclic garbage frozen
there instead would stay allocated through that sizing. Dropping it is safe only
if no other route or user code leaves device memory in cyclic garbage. The
2026-10-01 measurement and the cycle fix are recorded in the section 6b note of
[`DESIGN_path_tracer_roadmap.md`](algan/rendering/raytracing/DESIGN_path_tracer_roadmap.md).

**Work.** Record the warm CPU-only A/B for `b284532`. Then decide the
pre-render collection with evidence from more than one route (path tracer,
camera views, OOM retries): a young-generation collection is enough only if
the full one finds no tensor storage on any of them.

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
