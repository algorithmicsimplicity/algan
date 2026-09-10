# Renderer work queue

Current-status index, checked against `master` at
`f10cc230a108863d02980fc27079254473ae7de3` on 2026-09-09.
[TODO.md](TODO.md) is the prioritized backlog. This file preserves the original
item numbers so references from design documents and code remain meaningful.

The earlier queue mixed an August audit, completed implementation reports and
live recommendations. In particular, its red-CI notice, missing-material ports,
unreachable path tracer and inaccessible-settings claims were no longer current.
Those superseded narratives remain in Git history; they are not instructions to
reimplement features or regenerate baselines today.

## Status by original item

| Item | Current status | Evidence / remaining action |
| --- | --- | --- |
| 1. Truncation reporting | Implemented | `truncation.py` records render-job counters and warnings; retries are budgeting events, not silently dropped work. |
| 2. Missing verification harnesses | Historical inventory, not a current count | Many harnesses now exist; some old diagnostic names were retired. Check the named file and test, not the old count of 56. |
| 3. Identity-based shadow rejection | Implemented, default on | Identity-aware shadow acceptance helpers, `test_shadow_identity_epsilon.py`; not a cure for the geometric shadow terminator. |
| 4. Texture minification | Open | See the footprint/filtering work below and TODO item 1. |
| 5. Nested IOR | Implemented, default on | `nested_ior`, IOR stack code and `test_nested_ior.py`; the earlier claim that every interface assumes air is obsolete. |
| 6. Unlit planar circuits | Deliberate boundary | `TriangulatedBezierCircuit` provides a lit triangle alternative; a native lit-circuit path needs a product decision. |
| 7. Toon/normal/matcap/depth fragment ports | Implemented | Built-in pipeline IDs 0–9 in `shading_taichi.py`. Custom vertex-only shaders still have their own limitations. |
| 8. Inert settings and the old unwired path tracer | Removed/replaced | `light_intensity`/`ambient_light` inert fields and the old physical kernel were deleted. Current path tracing uses `path_tracer.py` and `path_tracer_taichi.py`. |
| 9. Resolve twice for shadowed sheets | Still a design tradeoff | Event/shade passes share transport; the optional `sheet_resolve_memo` arm exists and is off by default. Profile current end-to-end cost before changing it. |
| 10. Timeline query preparation cost | Optimized since the audit | Current row-query path and caches replace the old untargeted-work claim. Further work needs a fresh profile. |
| 11. Sparse discovery and sorting | Several optimized paths implemented | Fused compaction, packed keys, rank groups and MPS pixel sorting have separate gates. General `device_radix_sort` is not the same setting as `pixel_sort`. |
| 12. Batched geometry builds | Multiple optimizations implemented | See `agent_guidance/mobs_geometry.md`; profile the actual preparation stage and test geometry/AA output. |
| 13. CPU reclamation | Still worth measuring | The current function is `release_torch_memory`, not Algan's removed `empty_cache`; MPS has telemetry, CPU-only fallback still reports pressure. TODO item 6. |
| 14. Dead render experiments | Some remain | Legacy bloom helpers and unwired SMAA need caller-checked code cleanup. TODO item 7. |
| 15. Stale renderer documentation | Audited in this change | Module descriptions now distinguish sheet/wavefront/path transport, per-fragment defaults, texture support and backend-specific behavior. |
| 16. Inaccessible experimental fields | Old mapping defect resolved | `raytracing_settings.py` discovers storage modules and rejects writes to initialization-only fields deliberately. |
| 17. CPU baseline debt | Old failure report, not live CI status | Validate the exact SHA/backend/baseline key. Do not rebaseline from an old report or a missing-tool mismatch. |
| 18. Missing tracked implementation file | Historical incident | Test an installed wheel/source archive and required files; clean-checkout/package validation belongs in the release gates. |
| 19. Other design items | Mixed; inspect individually | Closed-shell continuations and crossing AA remain; old default-shader and retired fragment-walk tasks are not current work. |
| 20. Shadow-terminator offset | Implemented, default on | `shadow_terminator`, `_shadow_terminator_offset` and `test_shadow_terminator.py`; retain flat-geometry and smooth-surface controls. |

Paths in the evidence column are under `algan/rendering/raytracing/` or
`tests/unit_tests/` unless a directory is named explicitly.

## 4. Texture minification has no filter

Bilinear sampling reconstructs nearby texels but does not average a many-texel
footprint. Analytic geometric coverage does not solve that aliasing. The shared
UV samplers need a texture pyramid and a footprint estimate that survives both
raster and ray-traced hits. A small projected sheet area does not by itself say
how many UV texels the hit covers.

The screen-space glossy pyramid in
[DESIGN_glossy_prefilter.md](algan/rendering/raytracing/DESIGN_glossy_prefilter.md)
provides useful construction and sampling techniques, but is a different buffer
with a different level-selection problem. Preserve opacity, color-space and
normal-map semantics when adapting the techniques. Acceptance criteria are in
[TODO item 1](TODO.md#1-filter-minified-textures-on-primary-and-secondary-hits).

## Continuing correctness work

Per-sample sheet depth ownership is already implemented. What remains is a
better blend at within-pixel surface crossings, not the first implementation of
a sample depth buffer. Primary shell opacity and path-tracer shell handling also
do not automatically establish a common closed-solid contract for deterministic
reflected rays. See TODO items 2–3 and the detailed
[sheet](algan/rendering/raytracing/DESIGN_sheet_resolve.md) and
[mesh-identity](algan/rendering/raytracing/DESIGN_mesh_identity_open.md) designs.

The path tracer already supports rough glass with coupled multiple-scattering
compensation, homogeneous media, subsurface scattering, finite-light tree
sampling and denoising. Physical area geometry is implemented and its emitter
radiance is now receiver-independent, so neither the glass energy loss nor the
legacy radiance/falloff convention remains open work.

## Performance measurement rules

Use the active guidance in [agent_guidance/memory_perf.md](agent_guidance/memory_perf.md)
and [agent_guidance/gpu_harnesses.md](agent_guidance/gpu_harnesses.md). A Python
stage's inclusive time can contain synchronization for work launched earlier;
it is not the kernel's duration. Keep cold startup, warm cache loading, host
preparation, device execution and output encoding separate.

Compare both arms on the same compiler, backend and scene. Confirm the gate
actually engaged, measure its natural run-to-run output floor, and test video
windows as well as stills. MPS float32 reductions and split-pixel accumulation
do not have a universal cross-run/cross-device byte-identity guarantee.

Historical profiles are retained under `benchmarks/performance/reports/`.
Their improvements and rejected experiments remain useful evidence for those
runs, not current rankings or promises of a comparable gain on another GPU.

## Memory accounting

The render arena holds prepared geometry/acceleration data, route-specific hit
and path state, sheet/event buffers, frame accumulators and post-processing
scratch. Host preparation, compiler state, other PyTorch allocations and driver
reservations add costs outside that arena. Formats and live routes determine
the sizes; an old bytes-per-triangle table is not a universal allocation model.

`memory_model.py` fits recent arena peaks against chunk size with safety and
growth limits. Scene preflight and per-slot budgets still have explicit bounds.
OOM retry and overflow reporting remain necessary: a fixed per-path state size
or a successful first frame does not prove that a later frame will fit.
