# Deterministic renderer code audit — implementation status

Updated: September 11, 2026. Branch: `codex/renderer-audit-memory-cleanup`.
Pull request: #130, targeting `master` (not merged).

The original audit examined `f2073d718364617b35ed86028efefcd0ccda5606`.
The first implementation is `0b20f817ac25e4fe0aba9f9816b8485c9f26e8ec`.
The second implementation is `60597722f0e472ede8a173251f54282694172371`.
This document accompanies the third implementation, directly on that second
commit. Checked items describe code present on this branch, not a promise that
all platforms or the full test suite have passed.

Legend: `[x]` implemented; `[ ]` remaining. A **Partial** heading means the
listed completed sub-items are present but the entire audit recommendation is
not finished. “Unchanged” means a contract was preserved, not newly implemented.

## Summary of this update

The third tranche adds stage-scoped compaction workspace, shares the primary and
deferred shadow trace launch through a named context, resolves one immutable
prepared-batch execution policy, and makes the entire sparse attempt lifetime
exception-safe while preserving late-built BVHs. It also fixes the retry case
where BVH construction finishes but its arena publication fails.

The changes preserve accumulation/rounding boundaries, ordering, analytic
coverage and the existing conflict-rank ceiling. They complete substantial
parts of scratch ownership, but do not move every reduction result or tensor
expression into the arena and do not establish a measured speedup or peak-memory
reduction. The broad lifetime-region split and shared run CSR remain unfinished.

## 1. Arena-binding cache layout validation — Complete

- [x] Include tensor strides alongside pointers, storage, dtype, shape and device.
- [x] Reject the noncontiguous transpose after warming the cache with its contiguous square view.
- [x] Keep cache keys free of tensor references and rebuild whole-storage views per launch.

Implemented in the first tranche. Regression coverage is in
`tests/unit_tests/test_arena_args.py`.

## 2. Raw-fragment lifetime after sheet construction — Complete

- [x] Keep normal-render raw fragments and raw CSR in temporary discovery storage.
- [x] Retain only covered-pixel indices and resolver-ready sheets during shading/draining.
- [x] Preserve explicit raw-fragment retention for standalone diagnostics and armed capture.
- [x] Test one/two-frame renders after overwriting reclaimed forward scratch.

The first tranche releases 32 bytes per retained raw fragment plus raw CSR after
compaction. This is a lifetime saving, not an equivalent measured reduction in
discovery peak memory. The second tranche additionally avoids the final sheet
payload's allocator-to-arena copy.

## 3. Explicit compaction scratch and destinations — Partial

- [x] Give production compaction an explicit `resolver_memory` destination contract.
- [x] Return named `SheetBuffers` directly in persistent reverse-arena storage.
- [x] Gather nearest-depth and dominant-shading fields into their final destinations in one exact-copy kernel.
- [x] Build final int32 sheet offsets directly into the persistent CSR buffer.
- [x] Allocate native per-pixel and final-walk permutation outputs in forward arena scratch.
- [x] Preserve the standalone diagnostic/reference representation and compare its outputs against arena results.
- [x] Reserve the new arena permutation storage conservatively in discovery-footprint accounting.
- [x] Add `CompactionWorkspace` with nested, exception-safe forward scratch stages and an ordinary-allocator fallback for standalone callers.
- [x] Stage band accumulation scratch, conflict-rank lane scans and rank-group counts/CSR, shell-ceiling prefixes, nearest/dominant-reference scratch, sibling counts, lane-depth tables and lost-lane masks.
- [x] Accept caller-owned conflict-rank and lane-depth outputs; reject metadata mismatches and overlapping input/output byte ranges before writing.
- [x] Reuse short-lived scratch between consumers rather than retaining all converted temporaries until compaction ends.
- [x] Keep float64-to-float32 reduction boundaries and the float32 compatibility arm; always copy same-dtype shell scratch to avoid mutating source coverage.
- [x] Account for maximum overlapping workspace storage, including alignment, rather than summing disjoint stages.
- [ ] Move remaining long-lived reduction results, sorting/grouping intermediates and tensor expressions into explicit destinations or further staged workspace.
- [ ] Evaluate broader fused finalization only after preserving the current accumulation/rounding boundaries.

`sheet_buffers.py` and `sheet_output_taichi.py` own persistent resolver output.
`sheet_workspace.py` owns staged scratch; `sheet_statistics.py` names the
nearest/dominant-reference results. Long-lived reduction outputs, remaining
expressions, and library sort/scan workspace are still external allocations.
The workspace counter is not a total-device peak-memory measurement.

## 4. Unused diagnostic work — Complete for the audited outputs

- [x] Make compaction diagnostics optional, preserving the diagnostic default for standalone callers.
- [x] Skip the separate group-statistics pass and unnecessary final diagnostic gathers in normal rendering.
- [x] Keep truncation and exhausted-capacity reporting unconditional.
- [x] Specialize native band reduction to skip unused duplicate-lane diagnostics while preserving required area and union reductions.
- [x] Specialize native reference-statistics reduction to skip unused fragment counts; omit diagnostic return buffers in the reference arm as well.

Normal rendering still computes every statistic needed for coverage and
reference selection. Skipping optional diagnostics does not remove correctness
or truncation reporting.

## 5. Shared shadow submission and payload copies — Partial

- [x] Share a named shadow payload and exact gathering kernel between primary sheets and deferred secondary events.
- [x] Gather payloads into caller-scoped arena storage.
- [x] Scatter visibility directly into the already-initialized padded destination.
- [x] Preserve source identity, sorting, emitter sampling and optional footprint/terminator policies at their existing sites.
- [x] Share a frozen `ShadowTraceContext` and one explicit 48-argument shadow kernel launch across primary and deferred submission.
- [x] Pass current BVH objects at each launch and derive the refit template from the selected tree; do not cache a placeholder tree in the context.
- [x] Keep primary/deferred presence flags, identity rejection, sampling, footprint, sorting and terminator choices explicit at their original policy sites.
- [ ] Benchmark indexed tracing versus payload gathering before selecting another data-flow change.

Copy ownership and shared trace submission are implemented. The shared context
owns the single literal `raster_shadow_trace` call; the two event queues retain
their distinct preparation and sampling policies. Indexed tracing versus
payload gathering remains an unmeasured design choice.

## 6. Integer count–scan–write — Partial

- [x] Add `array_ops.csr_offsets` with caller-owned output, a terminal total and metadata/overlap validation.
- [x] Share raster count and acceptance-mask buffers across count kernels.
- [x] Remove the raster count concatenation, redundant whole-array conversion, exclusive-prefix subtraction and separate total sum.
- [x] Read write boundaries together and check capacity before narrowing to int32.
- [x] Use terminal CSR offsets in reference candidate expansion and native conflict-rank grouping, reusing the integer scan contract already used by native candidate expansion.
- [ ] Apply the contract to other independent integer scan sites where it removes duplicated work.
- [ ] Reuse an established CSR between consumers, rebuilding it only after stream-changing compaction/truncation.

Floating-point shell prefixes were not replaced by this integer helper.

## 7. Integer addresses versus float metadata — Complete

- [x] Introduce fixed float32 and int32 render metadata layouts.
- [x] Keep environment texel offsets/dimensions, bounce counts and glossy accumulator offsets in int32 from host preparation to kernel use.
- [x] Keep layer-order scalar, environment intensity and clipping distance in float32.
- [x] Initialize all fields on every route; remove optional-length decoding and float rounding biases for integer fields.
- [x] Validate host integer fields before allocation.
- [x] Test representability boundaries, including `2**23 + 1`, `2**24 + 1`, and the signed-int32 maximum without allocating huge textures.

`render_metadata.py` owns the layouts. The existing float/int arena buffers
carry both arrays, so the change does not add a separately bound Metal buffer.
Large-address tests prove the representation and binding contract; they do not
claim an actual multi-gigabyte texture render was performed.

## 8. ManualMemory hardening — Complete

- [x] Use `copy_` for scalar-safe clone/cast.
- [x] Validate integer, nonnegative dimensions before allocation arithmetic.
- [x] Construct/poison the typed view before committing pointer, high-water or recorder state.
- [x] Cover scalar, empty, alignment, reverse allocation and allocation-failure behavior.

Implemented in the first tranche. The second tranche does not change the
allocator's byte layout or reset semantics.

## 9. Named host records — Partial

- [x] `RayState` names the existing eleven-element tuple while retaining tuple unpacking and placeholder aliases.
- [x] `SheetBuffers` distinguishes resolver weights from the diagnostic raw-area representation.
- [x] `RenderMetadata` separates floating and integer payloads.
- [x] A named shadow payload records the shared event-copy layout.
- [x] `TriangleBounds` records immutable extrema and their tensor-precision diagonal.
- [x] `SheetStatistics` names nearest/dominant-reference results that survive scratch-stage boundaries.
- [x] `BatchExecutionPolicy` and `WavefrontPolicy` record tensor-free batch execution and allocation decisions.
- [x] `ShadowTraceContext` owns shared scene/light shadow launch context.
- [ ] Replace remaining large, string-keyed scene/fragment contracts with appropriate named records.
- [ ] Extend named context ownership beyond shadow submission and introduce an explicitly invalidated/shared run-CSR record.

Integer ray-state column 4 remains the sparse accumulator index; it was not
removed as “legacy padding.” Kernel interfaces still receive arrays/scalars,
not dynamic Python records per fragment.

## 10. Arena ABI schema and generation — Complete

- [x] Store original argument order and bound-array dtype/rank once in `arena_layouts.py`.
- [x] Generate literal binding prologues, arena specs and public parameter lists for all five packed kernels (four modules).
- [x] Validate kept kernel signatures before writing any generated files.
- [x] Provide `python scripts/generate_arena_bindings.py --check` and regeneration regression coverage.
- [x] Preserve independent prologue/spec, wrapper/signature, Metal buffer-budget and explicit launch-arity tests.
- [x] Keep ordinary annotated signatures and their comments inspectable; no runtime generation or unchecked launch splats were introduced.

The hot/cold array split remains unchanged. The first generation update also
carries the new integer metadata field through the two shade-kernel schemas.

## 11. One resolved batch execution policy — Complete for batch-level decisions

- [x] Resolve primary route, effective AA, shadow capability mode, fragment/custom-scatter requirements, continuation/IOR state width, pool ratio and core specialization choices into immutable prepared-batch records.
- [x] Reuse the policy across preflight, frame allocation, capability validation and wavefront execution.
- [x] Distinguish frame scale from in-place sample AA, so preflight does not charge a supersampled frame when execution allocates an output-resolution frame.
- [x] Expose primary route, effective AA and explicit fallback reasons through `RenderPlan`.
- [x] Clear the policy with its prepared/device scene and resolve live settings for a new render; test a real sheet-to-classic transition between renders.
- [x] Test route vetoes, all shadow capability modes, path-tracer shadow requests, and nested-IOR width decisions.

Concrete BVH objects remain launch-time data because deferred publication can
replace them. Defensive route/data consistency checks remain. Independent
microkernel optimization gates still have their own readers; concurrent settings
mutation during a running batch is not claimed to be supported.

## 12. Shared CSR and independent optimization gates — Partial

- [x] Decouple sheet-CSR construction from the diagnostic `sheet_metadata_kernel` switch.
- [x] Use lower bounds for standalone sheet offsets and direct destination writes for production offsets.
- [ ] Share unchanged run starts/counts among opaque-prefix, one-mesh and raw-fragment consumers.
- [ ] Evaluate batched endpoint downloads versus the current full host sheet-offset cache under adaptive retries; do not replace one transfer with many synchronizing reads without measurement.

## 13. Shared bounds construction and immutable scene facts — Complete

- [x] Share the triangle/circuit extents-to-bounds finishing operation.
- [x] Keep geometry-specific projection, clipping, behind-camera checks and opacity uncertainty outside that helper.
- [x] Preserve inclusive margins, frame shape, class flags and allocation direction.
- [x] Cache one triangle-extrema/diagonal record for shadow epsilon and Morton quantization.
- [x] Keep the original tensor-precision norm, triangle geometry set, and empty/degenerate/nonfinite shadow fallbacks.
- [x] Test bounds classification, persistence, cached facts and the old shadow-scale arithmetic.

This is shared preparation and metadata reuse, not a claim that all screen-bound
temporaries now live in the arena.

## 14. Lifetime classes and cleanup — Partial

- [x] Reclaim normal raw-fragment scratch at the discovery boundary.
- [x] Correct sparse allocation-failure retries to shrink the pool as well as the primary count; keep overflow retries' pool policy distinct.
- [x] Scope classic tile allocation, drain, allocator readback and compositing so every exit restores both arena ends.
- [x] Test success/reuse and exceptions injected during drain, readback and compositing, including non-OOM exceptions.
- [x] Scope each sparse attempt through allocation, resolve, drain, readback and compositing, restoring both arena ends on all exits subject to the retained batch floor.
- [x] Preserve and test late-built BVH bytes while poisoning reclaimed forward/reverse scratch after injected exceptions.
- [x] Separate BVH construction from arena-publication state: `bvh_rehome_pending` makes a smaller retry finish a failed arena copy after construction cleared `bvh_deferred`.
- [x] Test allocation failure during partial BVH publication and after completed publication; compare retry output with the unfailed render.
- [ ] Separate batch/chunk/tile/iteration lifetime regions where one reverse-stack retention floor otherwise retains intervening temporary allocations.

The retained reverse floor remains. The tests deliberately force a deferred
eligibility false positive to exercise runtime late construction; they do not
claim the reflective fixture would normally be selected for deferral. These
structured scopes fix cleanup/retry correctness but do not eliminate all
intervening reverse allocations retained beneath a late-built batch BVH.

## 15. Conflict-rank packing ceiling — Remaining, separate behavior change

- [ ] Replace all `parent * 16 + rank` assumptions with collision-free grouping.
- [ ] Remove the clamp only after reference grouping, native grouping and rank pooling all support larger ranks.
- [ ] Test same-surface overlaps at 16, 17 and substantially larger layer counts.

The current rank limit and its truncation reporting are unchanged. No claim of
unbounded same-surface transparency was added by this branch.

## 16. Helper ownership and historical clutter — Partial

- [x] Consolidate final bounds packing, immutable extrema, sheet finalization, shadow copying and metadata layouts into focused helpers.
- [x] Share the fixed full-union dust threshold between the host oracle and kernel reader rather than redefining it.
- [x] Correct the ray-state column-4 and classic compaction/lifetime descriptions touched by this work.
- [x] Update renderer guidance for schema-driven bindings rather than manual index renumbering.
- [x] Document staged workspace, prepared-batch policy, shared shadow launch and deferred-publication lifetime contracts in `DESIGN_memory_ownership.md`.
- [ ] Continue removing obsolete comments and remaining unsupported-path compatibility plumbing only after proving the callers and diagnostic fixtures no longer require it.

## Validation

### Third-tranche validation in this container

- [x] Expanded focused regression suite: **546 passed, 38 skipped** across
  31 modules. Coverage includes staged scratch poisoning/reuse, native/reference
  reduction parity, explicit output validation, shadow launch bindings, prepared
  policy reuse, and injected sparse-attempt/BVH-publication failures.
- [x] After correcting accidental fast markers on feature-only tests, reran the
  three changed feature modules plus fast-suite curation: **81 passed**.
  The standalone curation audit also passed both tests. These are overlapping
  reruns, not additional unique coverage; no feature tests were removed.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks passed; **419 files**
  were already formatted. Taichi kernel files were not formatted.
- [x] Generated bindings check passed for all four modules/five kernels;
  changed Python syntax checks and `git diff --check` passed.
- [x] Fast-render parity against unchanged `60597722`: **45 frames, 704 x 396,
  37,635,840 RGB values; maximum difference 0, differing values 0**.
  Both decoded RGB streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Six explicit two-frame route fixtures match that base byte-for-byte:
  sheet environment mapping, sheet glossy reflections, classic fused generation,
  classic supersampling, classic in-place AA, and a four-sample path-tracer
  fixture. Each covers **3,840 RGB values** with maximum difference 0. Both
  processes asserted their source path, with daemon handoff disabled.
- [ ] The final canonical fast suite is **not fully green**: **607 passed,
  1 failed, 3897 deselected**. The only failure is the known container
  MathTex/dvisvgm SVG-grouping baseline mismatch (maximum deviation 221 at
  frame 4). A fresh unchanged-base run produced **607 passed, the same failure**;
  the exact base-versus-new output comparison above isolates it from this change.
- [ ] The full `pytest -q` attempt reached its **600-second limit (exit 124)**
  before a final summary. It exposed three TeX-authoring example failures, each
  separately reproduced on unchanged `60597722` and on the new code:
  `text_and_math.rst:115`, `text_and_math.rst:190`, and `mob_gallery.rst:329` in
  `test_doc_examples.py`. Their missing SVG groups/indexing failures are not
  renderer regressions. It also exposed the fast-membership mismatch corrected
  above. The full suite was not rerun to completion after that marker-only fix.
- [ ] CUDA, Metal and AMD runtime validation; no cross-device pass is claimed.
- [ ] Alternating warm performance and actual peak-memory measurements.

The PR remains draft. No rendering baselines were changed. The 31-module run,
feature reruns and fast suite overlap, and are not a count of unique passing
checks. Scratch accounting is not a measurement of total device memory.

### Second-tranche validation (historical, recorded separately)

- [x] Final focused regression suite: **311 passed** across 17 modules. This
  includes exact metadata boundaries, code generation, independent arena ABI
  checks, sheet output parity, allocation/capture lifetimes, shadow copies,
  and injected retry/cleanup failures.
- [x] Updated weight-floor feature module: **4 passed**, including the real
  rendered gate variants. Its explicit argument-count/position pins were
  updated for the new integer metadata array, not removed.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks passed; **414 files**
  were already formatted. Taichi kernel files were not formatted.
- [x] Generated bindings check passed for all four modules/five kernels.
- [x] Changed Python syntax checks and `git diff --check` passed.
- [x] Fast-render parity against the unchanged previous branch commit:
  **45 frames, 704 x 396, 37,635,840 RGB values; maximum difference 0,
  differing values 0**. Both decoded RGB streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Four explicit two-frame route fixtures match the previous commit
  byte-for-byte: sheet environment mapping, sheet glossy reflections,
  classic fused generation, and classic supersampling. Each comparison
  covers 7,680 RGB values with maximum difference 0.
- [ ] The canonical fast suite is **not fully green**: **607 passed, 1 failed,
  3833 deselected**. The only failure is the known container MathTex/dvisvgm
  SVG-grouping baseline mismatch (maximum deviation 221). A fresh run of the
  unchanged previous branch reproduced it: 606 passed, the same sole failure.
  The base-versus-new exact comparison above isolates this from the change.
- [ ] The full `pytest -q` run was attempted with a 600-second limit and
  **timed out (exit 124)** before a final result. This is not a full-suite pass.
- [ ] CUDA, Metal and AMD runtime validation; no cross-device pass is claimed.
- [ ] Warm alternating performance and actual peak-memory measurements.

The PR remains draft for broader validation. Focused runs overlap with the
feature and fast suites; their counts are not an aggregate number of unique
passing tests.

### First-tranche validation (recorded separately)

The first implementation recorded 172 focused passes and 606 fast passes with
one reproduced container baseline failure. Its unchanged-master comparison
was also byte-identical over 45 frames. A 900-second full-suite attempt timed
out. Those are historical first-tranche results, not extra second-tranche
coverage and not a claim that the full suite or GPU validation passed.

No baselines were changed. CUDA/Metal/AMD validation and warm alternating
performance measurements remain outstanding. Moving visible outputs into an
arena does not remove library/compiler/driver workspace or external headroom
requirements.

## Recommended next implementation order

1. Finish long-lived compaction destination ownership and remaining expressions,
   using the new staged workspace rather than extending scratch lifetimes.
2. Introduce shared run CSR with explicit stream-change invalidation; retain the
   current endpoint-download policy until adaptive-retry behavior is measured.
3. Split batch/chunk/tile/iteration retention regions so a late-built BVH does
   not keep intervening reverse temporaries alive. Preserve the new publication
   and exception tests while doing so.
4. Validate CUDA/Metal/AMD runtime behavior and perform alternating warm timing
   and actual memory measurements before making performance claims.
5. Treat the conflict-rank ceiling as a separate semantic/capacity change with
   deep same-surface transparency fixtures, not as another copy refactor.
