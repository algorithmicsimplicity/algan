# Renderer work queue

Current-status index, re-checked against `master` at
`7e2ffc02173a1f4120c40e51eb0f65d5d4f79cc2` on 2026-10-01 (first checked at
`f10cc230`, 2026-09-09; item 4 was re-checked on 2026-09-30 against the texture
anti-aliasing commit `7bcc19c`).
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
| 4. Texture minification | Implemented, default on | `texture_antialiasing`, `texture_mips.py`, `texture_mips_taichi.py`, `test_texture_antialiasing.py` and `test_texture_antialiasing_taichi.py`; isotropic, and environment maps are not filtered. See below. |
| 5. Nested IOR | Implemented, default on | `nested_ior`, IOR stack code and `test_nested_ior.py`; the earlier claim that every interface assumes air is obsolete. |
| 6. Unlit planar circuits | Deliberate boundary | `TriangulatedBezierCircuit` provides a lit triangle alternative; a native lit-circuit path needs a product decision. |
| 7. Toon/normal/matcap/depth fragment ports | Implemented | Built-in pipeline IDs 0–9 in `shading_taichi.py`. Custom vertex-only shaders still have their own limitations. |
| 8. Inert settings and the old unwired path tracer | Removed/replaced | `light_intensity`/`ambient_light` inert fields and the old physical kernel were deleted. Current path tracing uses `path_tracer.py` and `path_tracer_taichi.py`. |
| 9. Resolve twice for shadowed sheets | Still a design tradeoff | Event/shade passes share transport; the optional `sheet_resolve_memo` arm exists and is off by default. Profile current end-to-end cost before changing it. |
| 10. Timeline query preparation cost | Optimized since the audit | Current row-query path and caches replace the old untargeted-work claim. Further work needs a fresh profile. |
| 11. Sparse discovery and sorting | Several optimized paths implemented | Fused compaction, packed keys, rank groups and MPS pixel sorting have separate gates. The general device radix sort (`ALGAN_DEVICE_RADIX_SORT`, `device_sort.py`, default off) is not the same switch as the sheet route's `sheet_pixel_sort` (`ALGAN_SHEET_PIXEL_SORT`, default on). |
| 12. Batched geometry builds | Multiple optimizations implemented | See `agent_guidance/mobs_geometry.md`; profile the actual preparation stage and test geometry/AA output. |
| 13. CPU reclamation | Main gap fixed; measurement and one decision remain | The current function is `release_torch_memory`, not Algan's removed `empty_cache`. `b284532` stopped a CPU render reporting GPU pressure (`test_an_unpressured_cpu_reclaim_skips_gc`), and a finished render no longer leaves its arena in cyclic garbage (`test_a_finished_render_leaves_no_closure_cycle_holding_its_arena`). The warm A/B is unrecorded, and the pre-render full collection is load-bearing. TODO item 5. |
| 14. Dead render experiments | Some remain | Legacy bloom helpers and unwired SMAA need caller-checked code cleanup. TODO item 6. |
| 15. Stale renderer documentation | Audited 2026-09-09; `renderer_limitations.rst` re-checked 2026-10-01 | Module descriptions distinguish sheet/wavefront/path transport, per-fragment defaults, texture support and backend-specific behavior. The limitations page was brought up to date in `e70a334` (feature matrix, light-tree selection, scale-aware offsets, media, truncation counters). |
| 16. Inaccessible experimental fields | Old mapping defect resolved | `raytracing_settings.py` discovers storage modules and rejects writes to initialization-only fields deliberately. |
| 17. CPU baseline debt | Old failure report, not live CI status | Validate the exact SHA/backend/baseline key. Do not rebaseline from an old report or a missing-tool mismatch. Separately, `path_traced/cuda` has not been re-rendered since `37dcc1b` (2026-09-05), although later path-tracer kernel changes forced the CPU set's re-baseline in `7cbd375`; expect it to be stale, which is inferred from history, not observed on a T4. |
| 18. Missing tracked implementation file | Historical incident | Test an installed wheel/source archive and required files; clean-checkout/package validation belongs in the release gates. |
| 19. Other design items | Mixed; inspect individually | Closed-shell continuations landed (`7314aaf`; TODO item 2); crossing AA remains (TODO item 1). Old default-shader and retired fragment-walk tasks are not current work. |
| 20. Shadow-terminator offset | Implemented, default on | `shadow_terminator`, `_shadow_terminator_offset` and `test_shadow_terminator.py`; retain flat-geometry and smooth-surface controls. |

Paths in the evidence column are under `algan/rendering/raytracing/` or
`tests/unit_tests/` unless a directory is named explicitly.

## 4. Texture minification anti-aliasing

**Status: implemented, default on.** See
`algan/rendering/raytracing/DESIGN_texture_antialiasing.md` for the data layout,
footprint model, costs and limitations. `SETTINGS.raytracing.texture_antialiasing`
controls the next merged batch and restores legacy bilinear filtering when off.

UV colour, material and normal maps now have shared-bank mip pyramids with
trilinear sampling. Primary sheet hits, classic hybrid hits and path-tracer hits
use UV density and projected ray-cone width, rather than coverage area alone.
Accumulated camera-path distance also filters reflected rays without extra ray
state. Linear-light/coverage-aware filtering preserves transparent edges; odd
sizes, packed base textures, animated endpoints and per-frame opacity are covered.

`benchmarks/_texture_antialiasing.py` is the asset-free primary/mirror A/B;
`tests/unit_tests/test_texture_antialiasing_taichi.py` pins numerical behaviour.
The cheap isotropic footprint deliberately does not implement anisotropic taps,
curved-mirror/refractive focusing or environment-map anti-aliasing.

## Continuing correctness work

Per-sample sheet depth ownership is already implemented. What remains is a
better blend at within-pixel surface crossings, not the first implementation of
a sample depth buffer: TODO item 1 and the detailed
[sheet](algan/rendering/raytracing/DESIGN_sheet_resolve.md) design. The
closed-solid opacity contract for deterministic reflected rays has landed
(`7314aaf`, TODO item 2): the wavefront pairs shell crossings on every straight
segment, as primary sheet compositing and the path tracer already did. Its one
residual is that the memory-trim permutation is disabled for batches carrying
shell IDs. The design discussion is in
[mesh-identity](algan/rendering/raytracing/DESIGN_mesh_identity_open.md).

The path tracer already supports rough glass with coupled multiple-scattering
compensation, homogeneous media, subsurface scattering, finite-light tree
sampling and denoising. Physical area geometry is implemented and its emitter
radiance is now receiver-independent, so neither the glass energy loss nor the
legacy radiance/falloff convention remains open work. What does remain is in
[`DESIGN_path_tracer_roadmap.md`](algan/rendering/raytracing/DESIGN_path_tracer_roadmap.md),
re-checked on 2026-10-01.

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
