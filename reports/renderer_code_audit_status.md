# Deterministic renderer code audit — implementation status

Updated: September 11, 2026. Branch: `codex/renderer-audit-memory-cleanup`.
Pull request: #130, targeting `master` (not merged).

The original audit examined `f2073d718364617b35ed86028efefcd0ccda5606`.
The first implementation is `0b20f817ac25e4fe0aba9f9816b8485c9f26e8ec`.
This document accompanies the second implementation change, directly on that
commit. Checked items describe code present on this branch, not a promise that
all platforms or the full test suite have passed.

Legend: `[x]` implemented; `[ ]` remaining. A **Partial** heading means the
listed completed sub-items are present but the entire audit recommendation is
not finished. “Unchanged” means a contract was preserved, not newly implemented.

## Summary of this update

The second tranche separates integer render addresses/counts from float
metadata, generates arena bindings for all five packed kernels from one layout
schema, writes final sheet records and their CSR directly into persistent arena
storage, provides arena destinations for native sheet-sort permutations,
introduces named ray/sheet/metadata records, shares conservative screen-bound
packing and immutable scene extrema, and makes classic tile attempts
exception-safe through readback and compositing.

The largest remaining work is the rest of compaction's scratch/reduction
ownership, one resolved batch policy, shared shadow trace context, and explicit
batch/chunk/tile/iteration lifetime regions. The conflict-rank ceiling is
unchanged and remains a separate behavior-changing project.

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
- [ ] Propagate explicit workspace/output ownership through band reductions, conflict ranks, shell-ceiling stages, lane-depth tables, sibling arithmetic and remaining tensor expressions.
- [ ] Introduce short scratch-stage scopes or reusable workspace so converting temporaries does not retain all of them until compaction ends.
- [ ] Evaluate broader fused finalization only after preserving the current accumulation/rounding boundaries.

`sheet_buffers.py`, `sheet_output_taichi.py` and the `resolver_memory` arm in
`sheets.py` implement the completed portion. Library sort/scan workspace and
most existing compaction tensor expressions are still external allocations.

## 4. Unused diagnostic work — Implemented for separable work

- [x] Make compaction diagnostics optional, preserving the diagnostic default for standalone callers.
- [x] Skip the separate group-statistics pass and unnecessary final diagnostic gathers in normal rendering.
- [x] Keep truncation and exhausted-capacity reporting unconditional.
- [ ] Optionally suppress diagnostic-only outputs inside shared fused reductions when that can be done without changing their required rendering results.

The last optimization is small, unfinished work inside shared reductions; a
normal render is not being described as having zero diagnostic-related stores.

## 5. Shared shadow submission and payload copies — Partial

- [x] Share a named shadow payload and exact gathering kernel between primary sheets and deferred secondary events.
- [x] Gather payloads into caller-scoped arena storage.
- [x] Scatter visibility directly into the already-initialized padded destination.
- [x] Preserve source identity, sorting, emitter sampling and optional footprint/terminator policies at their existing sites.
- [ ] Consolidate the repeated scene/BVH/light trace context and submission orchestration.
- [ ] Benchmark indexed tracing versus payload gathering before selecting another data-flow change.

Copy ownership is implemented; the two explicit `raster_shadow_trace` launch
sites remain. Their separate policies were not collapsed to make them shorter.

## 6. Integer count–scan–write — Partial

- [x] Add `array_ops.csr_offsets` with caller-owned output, a terminal total and metadata/overlap validation.
- [x] Share raster count and acceptance-mask buffers across count kernels.
- [x] Remove the raster count concatenation, redundant whole-array conversion, exclusive-prefix subtraction and separate total sum.
- [x] Read write boundaries together and check capacity before narrowing to int32.
- [ ] Apply the same contract to remaining candidate expansion and other independent scan sites.
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
- [ ] Replace remaining large, string-keyed scene/fragment contracts with appropriate named records.
- [ ] Introduce a batch kernel context and an explicitly invalidated/shared run-CSR record.

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

## 11. One resolved batch execution policy — Remaining

- [ ] Consolidate primary route, effective AA, shadow capabilities, continuation policy, state width and specialization choices into an immutable prepared-batch policy.
- [ ] Reuse that policy across preflight/allocation and execution rather than re-deriving subsets of it.
- [ ] Expose route/fallback reasons through the render plan.
- [ ] Test changes between renders without binding settings at module import.

Existing live-settings behavior and defensive route/data consistency checks
were preserved. Typed metadata is not a substitute for this execution policy.

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
- [ ] Convert remaining manual sparse-attempt cleanup to a structured ownership interface.
- [ ] Separate batch/chunk/tile/iteration lifetime regions where one reverse-stack retention floor otherwise retains intervening temporary allocations.
- [ ] Preserve and explicitly test late-built BVH retention while changing those lifetime regions.

The retained reverse floor was not removed. Late-built BVH lifetime and the
manual sparse-loop paths need a separate refactor rather than a superficial
replacement of pointer-reset calls.

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
- [ ] Continue removing obsolete comments and remaining unsupported-path compatibility plumbing only after proving the callers and diagnostic fixtures no longer require it.

## Validation

### Second-tranche validation in this container

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

1. Finish staged compaction workspace propagation, starting with the largest
   remaining reductions and explicitly shortening each temporary's lifetime.
2. Introduce the resolved prepared-batch policy and use it consistently in
   preflight and execution while preserving settings changes between renders.
3. Consolidate shadow trace context and replace remaining manual sparse-attempt
   cleanup with an interface that understands retained batch BVHs.
4. Treat the conflict-rank ceiling as a separate semantic/capacity change with
   deep same-surface transparency fixtures, not part of a copy refactor.
