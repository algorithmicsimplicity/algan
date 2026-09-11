# Deterministic renderer code audit — implementation status

Updated: September 11, 2026. Target branch: `codex/renderer-audit-memory-cleanup`.
Existing pull request: #130, targeting `master` (draft, not merged).

**Ninth implementation tranche:** floating-point and reference compaction work
on immutable published base `a883caee3f3a9ddd0e8b32e5f594a4d129556334`.
The implementation commit carrying this report continues the same branch and
existing draft PR. This is memory-ownership work, not a new renderer algorithm,
rank-capacity change, measured speedup or GPU-validation claim.

The original audit examined `f2073d718364617b35ed86028efefcd0ccda5606`.
The first implementation is `0b20f817ac25e4fe0aba9f9816b8485c9f26e8ec`.
The second implementation is `60597722f0e472ede8a173251f54282694172371`.
The third implementation is `077d2660c585e2f73b727dfd82f371a828c595bd`.
The fourth implementation is `dd4bfadc52387ac6249878f2602d094d8349862e`.
The fifth implementation is `234ac00663aa54a1837c4a2d8441fe52a224697c`.
The sixth implementation is `28b9736fa035826582f505f3e29d93fd0788a596`.
The seventh implementation is `a222eca1468990b459528641c3a336a23b888f51`.
The eighth tranche separated its memory refactor from the rank-capacity change:

- `41925b87118d93c77e43e8e8d6503a491e9c6149`: closed-shell ownership and lifetime refactor.
- `9f87cca1134d607dbf7ff1e6b1bf60a455d85a85`: full conflict ranks and collision-free grouping.
- `5a12513ca62ef09e70106fa71a572ebdd0c6adba`: explicit analytic-sheet route assertions in deep-render parity tests.
- `25ab7967cf2fe2befc8c460b68c0526b0561f83d`: cumulative eighth-tranche status.
- `a883caee3f3a9ddd0e8b32e5f594a4d129556334`: eighth-tranche publication record; the ninth tranche's exact base.

Checked items describe code published on this branch and are not a promise of
full-suite/GPU validation.

Legend: `[x]` implemented; `[ ]` remaining. A **Partial** heading means the
listed completed sub-items are present but the entire audit recommendation is
not finished. “Unchanged” means a contract was preserved, not newly implemented.

## Summary of this update

Geometry normal classification and primitive depth-slope tables now use
block-local arena scratch for frame gathers, normalization, cross products,
distance extrema, projection spans and class quantization. Their original
operations, input promotion and table-storage rounding boundaries are retained.
Each block reclaims its scratch before the next block; per-fragment depth-gap
work runs after table construction scratch is reclaimed.

Reference band reductions, coverage correction, nearest/dominant statistics,
sibling weights and lane-owner/depth conversion use explicit destinations and
reused predicates, index arrays and expression outputs. Sibling membership ends
before weight arithmetic. Float32 correction boundaries and the float32 round
before continuation signs (including negative zero) remain unchanged.

The expanded sample-depth reference has a checked caller-owned int32 loss-mask
output. Expanded input sorting, grouped surface minima, querying and final
comparison/packing use separate stages. The original best-depth/second-different-surface-depth
algorithm, strict epsilon and all-or-nothing cede test are preserved. Descriptor
capacity is reserved before sorting so input/group scratch can be released before
querying; this may increase arena residency for highly repeated keys and is not
in itself a measured peak reduction.

The shared tiny-floor helper accepts output/workspace arguments without changing
its default policy or autograd behavior. Tests caught an allocating vector-norm
wrapper despite its `out` argument; the direct vector-norm output API now owns that
result. Dispatch tracing verifies selected visible tensor outputs, not internal
library or driver workspace. New tests cover precision, strides, empty/scalar
inputs, destination validation, exact old-formula outputs, independent depth
oracles, block reuse, poisoning and exceptions.

No kernel source, packed ABI, rendering baseline or existing fast marker is
changed. Dynamic unique/nonzero results, library internals, reference conflict
prefixes and some optional boundary conversions remain external. Actual
CUDA/Metal/AMD execution and warm/peak-memory measurement are still outstanding.

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
- [x] Add checked caller-owned `BandComposite`, `BandReduction`, `SheetStatistics` and `SheetWeights` destinations, with ordinary-allocation defaults for standalone/reference callers.
- [x] Keep production band-composite results, final reductions/reference statistics, and sibling outputs in nested caller stages until the persistent resolver copy completes.
- [x] Reclaim rank-pooling area/union results before the next grouping operation.
- [x] Preserve empty/singleton output ownership and reject wrong metadata, input aliases, pairwise output aliases and disabled-output mismatches before mutation.
- [x] Add a checked caller-owned int64 stable lexsort destination; reuse per-pass gather, sort-value and composed-permutation scratch in nested stages.
- [x] Propagate caller-owned int32 outputs and staged key/count/scan workspace through optional device radix argsort and lexsort without changing their capability gates.
- [x] Allocate production pixel/final-walk permutations directly in forward scratch on every sort arm; stage shell-order permutations and gathered run keys.
- [x] Stage packed CUDA key/delta/sort-value storage while preserving the original queue-size, value-range and stable-sort policies.
- [x] Add exact caller-owned row gathers, with metadata/alias checks and local MPS-friendly integer copying; retain the exact allocating fallback where a local kernel is unavailable.
- [x] Release the rank-pooling per-fragment gather immediately after its reduction rather than retaining it through pooling-key construction.
- [x] Add `SortedFragments` with validated caller-owned pixel/depth/coverage/mask destinations and exact row copying; retain a private coverage copy for the shell ceiling.
- [x] Store production sorted payloads, rank/class inverse IDs and adopted group labels in forward workspace through their consumers and final persistent copy.
- [x] Stage packed class/rank keys and MPS pair-sort keys/boundaries/IDs, preserving the existing grouping and uniform-class reuse policies.
- [x] Share the class-group algorithm with `mps_compat.band_class_groups`; keep ordinary-owned standalone outputs and explicit native integer-width normalization.
- [x] Gather final per-sheet payloads and sample depths into caller storage, reuse final band IDs, and skip the unused representative permutation when the persistent copy is its only consumer.
- [x] Stage final mask-flag construction and sample-depth membership counts without changing signed sibling weights or any floating-point expression.
- [x] Rewind compaction's workspace on all exits, including failures after sorted-payload, rank and class generation; preserve prior forward sentinels and reverse outputs.
- [x] Add checked `FragmentMetadata` destinations and a shared exact frame/primitive gather, including packed depth bits and int32-to-int64 widening before address/key arithmetic.
- [x] Give shading-class and primitive-gap rules caller outputs; stage their frame tables and gathers while preserving the existing blocked floating-point expressions.
- [x] Reserve durable sort outputs before preprocessing scratch, gather raw shading classes immediately, and reclaim decoded metadata before rank grouping; test actual address reuse.
- [x] Stage the initial band-ID scan, rank-pooling maps, full-union/membership flags and pooling keys; retain a caller-owned inverse and the no-pooling `None` contract.
- [x] Reclaim pooling reductions before the second unique operation and transient class-split inputs before final reductions; keep production pool/class maps in caller storage.
- [x] Add checked `SampleDepthMetadata` and a scoped final depth-competition stage; preserve signed-weight, negative-zero, dust-threshold and mask-bit policies, including aliased production mask outputs.
- [x] Include worst-case alignment of both directly allocated int64 permutations in discovery accounting, separately from the staged workspace counter.
- [x] Test failure unwinding after metadata, shading-class, primitive-gap, pooling and sample-depth generation, in addition to earlier sorted/rank/class failure sites.
- [x] Add checked `ShellSegments` destinations; stage closed/active lookups, exact surface IDs, key arithmetic, facing flags and standalone contiguous copies.
- [x] Scope both native and reference closed-shell ceiling work, including group IDs, first positions, spent coverage, face sums, cap and scaling destinations; preserve the global scan and float32 rounding barriers.
- [x] End the complete closed-shell stage before rank grouping, validating actual pointer/stage restoration, scratch poisoning and failure cleanup.
- [x] Stage normal classification and primitive-slope floating-point blocks, with exact frame gathers, separate normalization/distance/projection lifetimes and unchanged promotion/table rounding.
- [x] Stage per-fragment primitive-gap scaling/threshold arithmetic after reclaiming table scratch.
- [x] Reuse reference band-reduction lane counts, union/sliver predicates and coverage-correction expression outputs, with checked popcount destinations.
- [x] Stage reference nearest/dominant statistics casts, positioned masks, maximum-coverage gathers, candidate masks and final exact gathers.
- [x] Reclaim sibling membership/run scratch before wide weight math; retain the float32 rounding before sign selection and ordinary no-multi aliases.
- [x] Reuse reference lane-owner predicates/indices/depth gathers; write native-owner/reference-depth conversion directly into its caller destination.
- [x] Add checked caller-owned expanded sample-depth loss masks and distinct sorting/grouping/query/packing lifetimes, preserving the original surface-minimum and cede algorithms.
- [x] Check actual visible operator destinations, block address reuse, reclaimed-memory poisoning, autograd predicate ownership and failures through the complete compaction attempt.
- [ ] Continue selected remaining integer-prefix, optional boundary/conversion and diagnostic temporary cleanup where justified. PyTorch unique/nonzero and sort/scan internals still allocate; adopting their results is not elimination of library workspace.
- [ ] Evaluate broader fused finalization only after preserving the current accumulation/rounding boundaries.

`sheet_buffers.py` and `sheet_output_taichi.py` own persistent resolver output.
`sheet_workspace.py` owns staged scratch and caller result lifetimes;
`sheet_reduction_buffers.py` and `sheet_statistics.py` name result destinations.
`sheet_order.py` owns stable-order composition scratch; `device_sort.py`
accepts native destinations. `array_ops.gather_rows` and `array_copy_taichi.py`
provide exact row-copy destinations. `sheet_fragments.py` names sorted payloads;
`sheet_grouping.py` owns shared grouping and inverse destinations.
`sheet_preprocessing.py` names decoded-fragment and sample-depth metadata;
`array_ops.gather_frame_table` owns exact scalar table lookup. `sheet_shells.py`
owns shell metadata and ceiling stages. `sheet_geometry.py` owns floating-point
geometry blocks; reference statistics, weights and expanded depth competition
now use explicit scratch stages. Dynamic unique/nonzero results, reference
conflict-prefix helpers, library workspace and optional index conversions still
require external storage.
Adopting a dynamic result into the arena does not eliminate its original
allocation. The workspace counter is not a total-device peak-memory measurement.

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
- [x] Add `group_ids_from_starts`, with caller-owned int32/int64 output, shape/dtype/device/overlap checks and an inclusive-scan capacity guard; remove separate boundary casts and subtraction results.
- [x] Apply it to initial bands, shell segments, diagnostic group counts, sample-depth enforcer groups and MPS pair-group boundaries.
- [x] Share `consecutive_pair_ids` between ordered rank-pooling descriptors and sorted MPS-friendly class pairs, with exact int64 destinations and staged boundary flags; raw unsorted conflict ranks still require sorting/grouping.
- [ ] Audit other independent integer scan sites outside these compaction paths; floating shell prefixes remain separate.
- [x] Reuse a `PixelRunCSR` between opaque-prefix, one-mesh and raw-fragment consumers; discard and rebuild it after stream-changing truncation.

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
- [x] `PixelRunCSR` names covered pixels, counts, offsets and fragment count for one immutable sorted stream, with explicit invalidation at membership changes.
- [x] `BandComposite`, `BandReduction` and `SheetWeights` distinguish reduction results from scratch and final resolver records.
- [x] `SortedFragments` names the sorted payload ownership contract; `RankGroups` names inverse IDs and parent/rank descriptors while preserving tuple unpacking.
- [x] `FragmentMetadata`, `RankPoolGroups` and `SampleDepthMetadata` name decoded stream facts, optional pooling maps and depth-competition eligibility without changing per-kernel argument layouts.
- [x] `ShellSegments` names exact sorted-stream shell keys/facing independently of short-lived reduction arrays.
- [ ] Extend named context ownership beyond shadow submission.

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
- [x] Share unchanged run starts/counts among opaque-prefix, one-mesh and raw-fragment consumers, independently of their kernel optimization gates.
- [x] Check stale count-record mixing without device reads; discard before filtering and rebuild on the new stream. In-place payload-only changes preserve the runs.
- [x] Use signed-int32 discovery offsets with a pre-scan capacity check; keep standalone int64 offsets and copy retained diagnostic offsets without rescanning.
- [x] Account for initial and replacement CSR storage, including alignment, conservatively in discovery headroom.
- [x] Test native/reference gates and real opaque/translucent discovery; require actual opaque truncation in the invalidation fixture.
- [ ] Evaluate batched endpoint downloads versus the current full host sheet-offset cache under adaptive retries; do not replace one transfer with many synchronizing reads without measurement.

The record is not a global cache or content hash. Its identity/count guard does
not detect arbitrary in-place tensor mutation; the caller must replace it after
any membership/order change, even with the same row count. Covered-pixel and
count tensors still use ordinary storage. Reuse does not alter the adaptive
retry endpoint-readback policy.

## 13. Shared bounds construction and immutable scene facts — Complete

- [x] Share the triangle/circuit extents-to-bounds finishing operation.
- [x] Keep geometry-specific projection, clipping, behind-camera checks and opacity uncertainty outside that helper.
- [x] Preserve inclusive margins, frame shape, class flags and allocation direction.
- [x] Cache one triangle-extrema/diagonal record for shadow epsilon and Morton quantization.
- [x] Keep the original tensor-precision norm, triangle geometry set, and empty/degenerate/nonfinite shadow fallbacks.
- [x] Test bounds classification, persistence, cached facts and the old shadow-scale arithmetic.

This is shared preparation and metadata reuse, not a claim that all screen-bound
temporaries now live in the arena.

## 14. Lifetime classes and cleanup — Complete for the audited renderer paths

- [x] Reclaim normal raw-fragment scratch at the discovery boundary.
- [x] Correct sparse allocation-failure retries to shrink the pool as well as the primary count; keep overflow retries' pool policy distinct.
- [x] Scope classic tile allocation, drain, allocator readback and compositing so every exit restores both arena ends.
- [x] Test success/reuse and exceptions injected during drain, readback and compositing, including non-OOM exceptions.
- [x] Scope each sparse attempt through allocation, resolve, drain, readback and compositing, restoring both arena ends on all exits subject to the retained batch floor.
- [x] Preserve and test late-built BVH bytes while poisoning reclaimed forward/reverse scratch after injected exceptions.
- [x] Separate BVH construction from arena-publication state: `bvh_rehome_pending` lets a retry finish a failed copy after construction cleared `bvh_deferred`.
- [x] Test allocation failure during partial BVH publication and after completed publication; compare retry output with the unfailed render.
- [x] Separate compaction band-composite, final-reduction and final-copy scratch/result stages; preserve persistent outputs across reclaimed forward storage.
- [x] Publish late-built BVHs only after unwinding the tile, coverage and chunk-output regions, eliminating the intervening reverse-allocation retention hole.
- [x] Restart/refill the discarded chunk and restore truncation/path-sample counters; accept no partial composites or discarded statistics.
- [x] Roll back both arena ends after partial publication, allow one allocator-reclaim retry, and escalate exhausted clean-boundary capacity to prepared-batch recovery; do not retry non-memory errors.
- [x] Test the actual publication pointers, discarded-background poisoning, one-time publication, later chunk splitting/reuse, and transactional bounded failures.
- [x] End raw preprocessing before rank grouping, pool reductions before regrouping, and sample-depth scratch before persistent output copying; verify reuse, poisoning and injected-failure unwinding.
- [x] Include shell eligibility, segment results and ceiling reduction in the preprocessing lifetime; unwind lookup/sort/scan/reduction/copy/kernel failures before subsequent stages, preserving sentinels and persistent output.

The retained-floor mechanism still protects genuine batch tables and trees;
this is not a redesign of `ManualMemory` as an arbitrary region allocator.
Current batch publishers run before coverage or at the clean restart boundary,
so they no longer pin coverage/tile allocations. Tests deliberately force a
false-positive deferral eligibility decision; ordinary reflective scenes are
not claimed to select deferral. That exceptional path pays one chunk replay,
not an eager BVH reservation for every otherwise split-free batch.

## 15. Conflict-rank packing ceiling — Implemented and published; GPU validation remaining

- [x] Replace shipping `parent * 16 + rank` assumptions in reference rank grouping and rank pooling with collision-free row-count-radix or two-field grouping.
- [x] Retain every derived conflict rank, removing the rank-15 clamp and its producer of truncation events only after native, reference and pooling paths support larger ranks.
- [x] Validate the host row count before compaction allocation. Dense producer IDs/ranks are less than that count, so signed-int32 capacity bounds packed keys below 2**62; arbitrary unrestricted int64 pairs are not covered by this proof.
- [x] Keep MPS-friendly fallback keys as separate bounded components, using sorting for raw ranks that may decrease and a no-sort adjacent-pair scan only for already ordered pooling descriptors.
- [x] Test same-surface overlaps at 16, 17, 64/65 and 257 crossings, donors, decreasing ranks, eligible/ineligible pooling and neighboring parents, native/reference arms and persistent/default output lifetimes.
- [x] Check sequential transparency against an independent oracle and compare actual 17-/65-layer same-surface images with equivalent independent-surface images.
- [x] Keep the legacy public `sheet_layers` field/warning formatter for report compatibility, while replacing the old clamp-reporting regression with one asserting actual deep ranks and no retired event. Active truncation counters are unchanged.
- [ ] Validate these changed paths on actual CUDA, Metal and AMD hardware and measure the additional capacity's performance/memory behavior.

This is a deliberate behavior change for streams formerly exceeding 16 layers,
not a claim of identical output for those old incorrect cases. Ordinary fixtures
below the former ceiling retain parent parity. There is no fixed per-surface
sheet-rank ceiling now; finite memory, signed-int32 row capacity and independent
ray/path-tracer limits remain. Historical benchmark fixtures containing four-bit
keys are not the shipping grouping implementation.

## 16. Helper ownership and historical clutter — Partial

- [x] Consolidate final bounds packing, immutable extrema, sheet finalization, shadow copying and metadata layouts into focused helpers.
- [x] Share the fixed full-union dust threshold between the host oracle and kernel reader rather than redefining it.
- [x] Correct the ray-state column-4 and classic compaction/lifetime descriptions touched by this work.
- [x] Update renderer guidance for schema-driven bindings rather than manual index renumbering.
- [x] Document staged workspace, prepared-batch policy, shared shadow launch and deferred-publication lifetime contracts in `DESIGN_memory_ownership.md`.
- [x] Document shared pixel-run invalidation and caller-owned result lifetimes, and correct earlier present-tense descriptions of external reduction output storage.
- [x] Correct deferred-publication/retry comments and document clean-boundary restarts, exact gathers and stable-sort output lifetimes.
- [x] Consolidate the compatibility class-group implementation and correct touched comments about external reduction/sorted-payload ownership.
- [x] Document preprocessing/table/pooling/sample-depth destination contracts and correct touched comments about metadata lifetimes and discovery workspace accounting.
- [x] Extract and document shell ownership, distinguish the new rank-capacity fix from the exact refactor, update user-facing renderer limits and retain the legacy report field explicitly rather than implying it still clamps layers.
- [x] Extract block geometry ownership, remove the retired private `_rows` wrapper, and document phase-local floating/reference buffers and the limits of dispatch-level allocation checks.
- [ ] Continue removing obsolete comments and remaining unsupported-path compatibility plumbing only after proving the callers and diagnostic fixtures no longer require it.

## Validation

### Ninth-tranche validation

- [x] Final focused regression suite: **1,410 passed, 27 skipped**, across
  **37 distinct modules**, on the final Python source. The two new modules
  contain **243 cases** (173 floating-workspace and 70 depth-reference cases);
  two additional full-compaction failure cases bring the tranche to **245 added
  tests**. These are included in the focused count, not extra unique coverage.
- [x] Exact old-formula class, slope, coverage and sibling-weight comparisons
  cover ordinary/MPS-friendly CPU policies, mixed float32/float64 inputs,
  frame wrapping, strided/scalar/empty inputs, negative zero and boundary values.
  Independent per-sheet depth oracles cover surface identity, missing lanes,
  strict depth epsilon, and all-or-nothing ceding. Scope tests verify actual
  block-address reuse, poisoning and exceptional exits; invalid output metadata
  and aliases are rejected before mutation. Runtime failures can partially write
  results, which the enclosing attempt discards.
- [x] Operator-dispatch tracing verifies selected visible fixed-size tensor
  results use input/caller/workspace storage. It exposed a norm wrapper that
  allocated despite `out`; the direct vector-norm output route now passes.
  Default autograd floor calls retain the saved predicate outside reusable
  scratch, including after poisoning. This is not a library/driver allocation
  measurement or proof that every operation is allocation-free.
- [x] Parent-versus-new compaction: **256 configurations, all bit-identical**,
  across 32 seeded one-to-three-frame inputs, native/reference, diagnostic/
  persistent and ordinary/MPS-friendly CPU policies. This round also explicitly
  disables band-statistics and depth-reduction kernels in reference runs, forces
  small geometry blocks, and supplies animated projection tables. Source paths
  are asserted and reclaimed storage is poisoned before serialization. Matching
  hashes for each parent/new pair:
  - ordinary: `78b41ecc1b72fee9637e6b70779ed2bda4a59e6741c9c9aeebc1ee44a253a3db`;
  - MPS-friendly CPU: `d3835dbab8328a16c8f3dcfc7e23b43d17df37c4fb416cee57969e8a9d9315a2`.
- [x] Decoded standard fast-render comparison against unchanged `a883caee`:
  **45 frames, 704 x 396, 37,635,840 RGB values; maximum difference 0,
  differing values 0**. Both streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Repository-wide Ruff 0.12.4 lint and formatting pass (**440 files**);
  all **eight changed Python files** parse; generated arena bindings and diff
  checks pass. No kernel source, packed ABI, baseline or fast marker changed.
- [ ] Canonical fast suite is **not fully green**: **608 passed, 1 failed,
  4835 deselected**. Unchanged parent: **608 passed, 1 failed, 4590 deselected**.
  Both fail only the known MathTex/dvisvgm baseline mismatch, maximum channel
  deviation 221 at frame 4. The decoded parent/new outputs match exactly.
  The final collection contains **5,444 tests**, 245 more than the parent.
- [ ] Full `pytest -q` reached its **600-second limit (exit 124)** without
  a final summary; its last printed progress was **27%**. The **3 observed
  failures** were:
  - `tests/unit_tests/test_doc_examples.py::test_doc_example_authors_without_error[advanced_user_tutorials/text_and_math.rst:115]`.
  - `tests/unit_tests/test_doc_examples.py::test_doc_example_authors_without_error[advanced_user_tutorials/text_and_math.rst:190]`.
  - `tests/unit_tests/test_doc_examples.py::test_doc_example_authors_without_error[galleries/mob_gallery.rst:329]`.
  Source-verified isolated runs reproduce these cases on both unchanged parent
  and new code; the results are not a completed full-suite/heavy-baseline pass.
- [ ] CUDA/Metal/AMD execution, alternating warm performance and actual
  total-device peak-memory measurement remain outstanding.

Validation used the supplied editable CPU installation: Python 3.13.5,
PyTorch 2.10.0+cpu and patched `quadrants` distribution 1.3.0.post1 (runtime 1.3.0).
Focused and renderer comparison runs disable daemon handoff; the full attempt
leaves `ALGAN_USE_DAEMON` unset. The final focused run supersedes the earlier
1,396-pass run, which preceded the last 14 feature cases. Suite and probe counts
overlap and are not an aggregate unique-test total. CPU execution under the
MPS-friendly policy is not actual Metal validation. No measured speedup or
reduction in total device peak memory is claimed.

### Eighth-tranche validation (historical)

- [x] Final focused suite: **1,111 passed, 27 skipped**, across **34 modules**.
  The two new modules contain 110 shell cases and 109 deep-rank cases (including
  two actual render comparisons). The pair-scan module adds 20 cases and the
  existing fragment-lifetime module adds four failure cases: **243 added tests**
  in total, included in the focused count rather than extra unique coverage.
- [x] Cover native/reference paths and actual ordinary/MPS-friendly CPU policies,
  exact frame-table IDs, empty/inactive/singleton/strided inputs, caller/default
  output ownership, atomic metadata/alias validation, float32/float64 rounding,
  negative zero, scratch poisoning and success/failure pointer restoration.
  Runtime exceptions can leave caller outputs partially written; enclosing
  attempts discard them. Scratch unwinding is not transactional result rollback.
- [x] Final parent/new comparison: **256 compaction configurations, all
  bit-identical**, across 32 seeds, native/reference, diagnostic/persistent and
  ordinary/MPS-friendly CPU policy arms. One-to-three-frame fixtures include
  time offsets, animated/static identities and declarations, closed shells,
  shading splits and sample-depth handling. Source imports are asserted and
  freed forward storage is poisoned before serialization. These fixtures stay
  below the former per-surface rank ceiling. Matching parent/new JSON hashes:
  - ordinary policy: `78b41ecc1b72fee9637e6b70779ed2bda4a59e6741c9c9aeebc1ee44a253a3db`;
  - MPS-friendly CPU policy: `d3835dbab8328a16c8f3dcfc7e23b43d17df37c4fb416cee57969e8a9d9315a2`.
- [x] Independent deep-stack pair/grouping and sequential transparency oracles
  retain all crossings through 257 layers. The actual 17-layer and 65-layer
  transparent same-surface render fixtures match their independent-surface
  references pixel-for-pixel and are nonblack; they are not vacuous count tests.
- [x] Repository-wide Ruff 0.12.4 lint/format checks pass (**437 files**);
  generated bindings and diff checks pass. No Taichi kernel source, packed ABI,
  rendering baseline or existing fast marker changed.

- [x] Decoded fast-render comparison with unchanged `a222eca1`: **45 frames,
  704 x 396, 37,635,840 RGB values; maximum difference 0, differing values 0**.
  Both streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [ ] Canonical fast suite is **not fully green**: **608 passed, 1 failed,
  4590 deselected**. Fresh unchanged parent: **608 passed, 1 failed,
  4347 deselected**. Both fail only the MathTex/dvisvgm compatibility baseline
  at frame 4, maximum channel deviation 221. The decoded new/parent images
  nevertheless match exactly. No baselines or existing fast markers changed.
- [x] Full collection contains **5,199 tests**, 243 more than the parent; new
  feature tests remain outside the curated fast suite.
- [x] After the broader suite, strengthened both deep-render fixtures to force
  deterministic SPP=1 and require `primary_route == "analytic_sheets"` for each
  same-surface and independent-surface render. Reran them: **2 passed,
  107 deselected**. These are the same two cases already counted above, not new
  tests. Production code did not change after the broader suite/probe runs.
- [x] Clean-checkout replay reconstructs all three local code/test commits
  exactly, including code tree `009666a00d3c9e91529813cb365f8c3f1e77b8b6`.
  All **12 changed Python files** parse; replayed generated bindings pass.

- [ ] Full `pytest -q` reached its **600-second limit (exit 124)** without a
  final summary; its last printed progress indicator was **29%**. The three
  observed failures were the TeX-authoring examples at
  `advanced_user_tutorials/text_and_math.rst:115`, `:190`, and
  `galleries/mob_gallery.rst:329`. The collected order and per-test log identify
  the same three cases. Source-verified isolated runs on both unchanged parent
  and new code reproduced all three with identical MathTex group/index errors.
  The complete suite and heavy-render baselines have not passed in this run.
- [ ] CUDA/Metal/AMD execution, alternating warm performance and actual
  total-device peak-memory measurements remain outstanding.

Validation uses the supplied editable CPU install: Python 3.13.5, PyTorch
2.10.0+cpu and patched `quadrants` distribution 1.3.0.post1 (runtime banner 1.3.0).
Renderer probes/focused/fast runs disable daemon handoff; the full attempt leaves
`ALGAN_USE_DAEMON` unset. Focused, fast, standalone render and probe results
intersect and are not a sum of unique tests. No actual GPU execution, alternating
warm benchmark, total-device peak measurement or completed heavy-baseline suite
is claimed. The changes are published on the existing PR branch; PR #130 remains
draft pending the broader validation described below.

### Seventh-tranche validation (historical; unchanged below)

- [x] Final focused regression suite: **855 passed, 28 skipped**, across
  **32 distinct modules**. The three new feature modules contain **207 cases**
  (110 preprocessing/table/sort-destination, 65 pooling and 32 sample-metadata
  cases). The existing fragment-lifetime module adds ten parametrized cases
  covering five additional failure sites. These 217 added cases are included
  in the focused total, not additional unique coverage.
- [x] Cover caller-owned/default outputs, native/reference gates, static and
  animated tables, wrapped frame rows, empty/singleton/strided inputs, exact
  large integers and packed depth bits, all-output validation before mutation,
  input and pairwise output aliases, signed weights and negative zero, and
  dust-threshold/mask policies. Tests exercise the MPS-friendly copy policy on
  CPU and its exact allocating fallback; this is not Metal execution.
- [x] Assert actual decoded-pixel storage reuse by rank grouping and poison
  reclaimed pooling scratch before its second unique. Success and injected
  failures after metadata, shading classes, primitive gaps, sorted payloads,
  ranks, pooling, classes and sample-depth metadata restore forward pointers
  and preserve earlier sentinels and reverse outputs.
- [x] Final parent-versus-new compaction comparison: **256 configurations,
  all bit-identical**, over 32 seeds with native/reference, diagnostic/persistent
  and ordinary/MPS-friendly CPU policy arms. Fixtures cover one to three frames,
  time offsets, static/animated surface tables, shell ceilings, shading splits,
  positioned depth and sample-depth handling. Each process asserts its imported
  source path; reclaimed forward storage is poisoned before serialization.
  Parent and final new JSON files have SHA-256
  `b6c2110fb3fde9b5111c035134cdda9c037fabaef8a46908e0d1509415f70032`.
- [x] Decoded fast-render parity against unchanged `28b9736`: **45 frames,
  704 x 396, 37,635,840 RGB values; maximum difference 0, differing values 0**.
  Both decoded streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks pass (**434 files**).
  All **nine changed Python files** parse, and generated arena bindings and
  `git diff --check` pass. No Taichi kernel source, packed ABI, rendering
  baseline or pre-existing fast-test marker changed.
- [ ] The canonical fast suite is **not fully green**: **608 passed, 1 failed,
  4347 deselected**. A fresh unchanged parent produced **608 passed, 1 failed,
  4130 deselected**. Both fail the same MathTex/dvisvgm compatibility baseline
  at frame 4, maximum channel deviation 221. The exact parent/new video
  comparison above isolates this mismatch from the implementation changes.
- [ ] Full `pytest -q` was attempted and reached its **600-second limit
  (exit 124)** without a final summary. Its last printed progress indicator was
  **30%**. Three observed failures map to the TeX authoring examples at
  `advanced_user_tutorials/text_and_math.rst:115`, `:190` and
  `galleries/mob_gallery.rst:329`. Each was reproduced
  independently on the unchanged parent and new code with the same missing
  MathTex SVG groups and indexing errors (three failures in each isolated run).
  Collection found 4,956 tests. This is not a full-suite pass or a completed
  heavy-render comparison.
- [ ] CUDA, Metal and AMD runtime validation, alternating warm performance and
  actual total-device peak-memory measurements remain outstanding.

The PR remains draft. Validation used Python 3.13.5, PyTorch 2.10.0+cpu and the
supplied patched Quadrants runtime 1.3.0 in an editable CPU installation. The
renderer probes and focused/fast runs disabled daemon handoff; the full attempt
left `ALGAN_USE_DAEMON` unset to avoid the previously documented CLI expectation
mismatch. Focused, fast and comparison counts overlap and must not be added
as unique-test totals. Intermediate seventh-tranche results are superseded by
these final results; earlier-tranche results below remain historical.

### Sixth-tranche validation (historical; unchanged below)

- [x] Final focused regression suite: **658 passed, 27 skipped**, across
  **26 distinct modules**. This includes **100 new grouping/fragment cases**
  (78 grouping and 22 fragment/lifetime cases), plus one host-pressure reclaim
  case. These are included in the focused count, not additional unique coverage.
- [x] Caller-owned/default outputs, native/reference and MPS-friendly policy
  arms, empty/singleton/strided inputs, input and pairwise output aliases,
  output-before-mutation validation, exact integer/depth bits, scratch poisoning,
  and success/failure forward-pointer restoration are covered. Three injected
  failure sites follow sorted-payload, rank and class generation.
- [x] Added regression coverage for int32 input keys whose products exceed int32:
  an int64 `out` does not widen PyTorch multiplication. Key builders now copy
  to int64 before multiplying. The normal int64 production path is unchanged.
- [x] Final parent-versus-new compaction comparison: **128 configurations, all
  bit-identical**, over 32 seeded inputs with native/reference and diagnostic/
  persistent arms, including closed shells, shading splits and sample-depth
  handling. Reclaimed forward storage is poisoned before output serialization;
  each process asserts its imported source path. Both JSON files have SHA-256
  `bd9dd5c2923b09defffd69edb508e839a9df7dda2e0bcf7be82e433343b5503a`.
- [x] Final fast-render parity against unchanged `234ac006`: **45 frames,
  704 x 396, 37,635,840 RGB values; maximum difference 0, differing values 0**.
  Both decoded RGB streams have SHA-256
  `6b197223d11672c001b3f7782362aca735a34a8b6c3497cce401a29010eb8e53`.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks pass (**430 files**).
  All **nine changed Python files** parse; generated bindings and
  `git diff --check` pass. No Taichi kernel source or packed ABI was modified.
- [x] Stabilized the pre-existing unpressured-MPS reclaim test by explicitly
  mocking host pressure as well as GPU pressure; added the complementary
  host-pressure case. The earlier broad run's failure was reproduced on the
  untouched parent with host pressure forced true. This is test isolation, not
  a change to memory-reclaim behavior. The final suite above passes afterward.
- [ ] The final canonical fast suite is **not fully green**: **608 passed,
  1 failed, 4130 deselected**. A fresh unchanged parent produced **608 passed,
  1 failed, 4029 deselected**. Both failures are the existing MathTex/dvisvgm
  baseline mismatch, maximum channel deviation 221 at frame 4. No baselines or
  existing fast-test markers were changed.
- [ ] Full `pytest -q` was attempted and reached its **600-second limit
  (exit 124)** without a final summary, at approximately 12% progress. Its one
  observed failure maps to
  `test_cli.py::test_a_plain_run_launches_the_script_as_its_own_process` in the
  collected order. The isolated test fails on both unchanged parent and new
  code because the validation environment sets `ALGAN_USE_DAEMON=0` while the
  test expects that variable to be absent. This is not a full-suite pass or a
  completed heavy-render comparison.
- [ ] CUDA, Metal and AMD runtime validation, alternating warm performance and
  actual total-device peak-memory measurements remain outstanding.

The PR remains draft. Validation used the supplied editable CPU installation
and disabled daemon handoff for renderer tests and comparison probes. CPU
execution under the MPS-friendly policy is not a Metal runtime pass. Focused,
fast and comparison runs overlap; their counts are not an aggregate unique-test
total. Intermediate sixth-tranche results are superseded by the final results
above. Earlier-tranche results below remain historical.

### Fifth-tranche validation (historical; unchanged below)

- [x] Final focused suite: **494 passed, 28 skipped** across **26 distinct
  modules**, after the final seven metadata/native-sort failure tests were added.
  Includes real sparse discovery/retry/capture fixtures, independent arena ABI
  checks, shared runs, compaction results, stable ordering and exact row copies.
- [x] The new lifetime module's **five tests** cover a forced late continuation
  with actual clean-boundary pointer assertions, discarded-background poisoning,
  truncation/path-sample rollback, one-time tree publication, post-publication
  chunk splitting, and bounded transactional publication failures. These five
  are included in the focused count, not additional coverage.
- [x] The new sort/gather module's **65 tests** cover caller-owned/default outputs,
  empty/singleton streams, stable ties, strided/nonfinite floating keys, large
  integer bits, invalid destinations, scratch poisoning/reuse, packed-key
  arithmetic, and injected nested-sort failures. The local exact-copy kernel was
  exercised on CPU under the MPS-friendly policy. CPU-native radix test doubles
  check the host storage/composition contract, not CUDA/Metal runtime behavior.
- [x] Parent-versus-new compaction comparison: **128 configurations**, all
  bit-identical. Native/reference and diagnostic/persistent outputs were checked
  across 32 seeded fixtures, including closed shells, shading splits, sample
  depths and nearest/dominant references. Reclaimed scratch was poisoned before
  serializing outputs, and each process asserted its imported source path.
  Both result JSON files have SHA-256
  `d221adbd6bff24115ce64aa105d88f1ce699ccc356bd9f0f00de3e53c6e468bf`.
- [x] Final fast-render comparison against unchanged `dd4bfadc`: **45 frames,
  704 x 396, 37,635,840 RGB values; zero differences**. Both decoded streams have
  SHA-256 `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks passed (**426 files**).
  All eleven changed Python files parse; generated bindings and
  `git diff --check` pass. The new Taichi copy kernel was linted, not formatted.
- [ ] The final canonical fast suite is **not fully green**: **608 passed,
  1 failed, 4029 deselected**. Unchanged `dd4bfadc` produced **607 passed,
  1 failed, 3959 deselected**. Both failures are the same MathTex/dvisvgm baseline
  mismatch, maximum deviation 221 at frame 4. The additional fast case is the
  existing automatic static-control-flow check discovering the new copy-kernel
  file; no existing test markers or rendering baselines were changed.
- [ ] Full `pytest -q` reached its **600-second limit (exit 124)** at approximately
  23% progress, without a final summary. Its one observed failure was
  `test_cli.py::test_a_plain_run_launches_the_script_as_its_own_process`: it
  expects `ALGAN_USE_DAEMON` to be absent, whereas this validation environment
  explicitly disables daemon handoff. The same isolated failure reproduced on
  both parent and new code; it passed on the parent after removing that override.
  The full suite was not rerun to completion in that alternate environment.
  No full-suite pass or completed heavy-baseline comparison is claimed.
- [ ] CUDA, Metal and AMD runtime validation, alternating warm performance and
  actual total-device peak-memory measurements remain outstanding.

The PR remains draft. CPU tests used the supplied editable-install environment,
with daemon handoff disabled and no baseline updates. The final focused, fast
and probe results overlap; their counts are not an aggregate unique-test total.
Earlier fifth-tranche intermediate runs are superseded by the final focused and
fast results above. Previous-tranche route fixtures below remain historical.

### Fourth-tranche validation (historical, unchanged below)

- [x] Final focused run: **453 passed, 28 skipped** across **22 distinct
  modules**, including the new shared-CSR and caller-owned-destination tests,
  real raster lifetime/capture/retry fixtures, native/reference reductions,
  independent arena bindings and batch policy tests.
- [x] Injected failures after band reduction, after reference selection and
  before the persistent final copy unwind every result stage. Persistent
  sentinels survive overwriting reclaimed forward storage. These three tests
  are included in the focused count, not additional unique coverage.
- [x] Exact parent-versus-new compaction comparison: **128 configurations**, all
  bit-identical. This covers native/reference gates and diagnostic/persistent
  outputs across 32 seeded fragment fixtures, including closed shells, shading
  splits, sample-depth handling and nearest/dominant references. Reclaimed
  arena storage is poisoned before serializing outputs. Both result JSON files
  have SHA-256
  `4a00a0c24062501de4992fc51b01714ef948ec2017ee3d476d5bd1e0e8ed4e22`.
- [x] Final fast-render parity against unchanged `077d2660`: **45 frames,
  704 x 396, 37,635,840 RGB values; maximum difference 0, differing values 0**.
  Both decoded RGB streams have SHA-256
  `65def67ad219ec774d71a5d4457d89dab1ab08f190a04e9a8d32bc544da6616a`.
- [x] Repository-wide Ruff 0.12.4 lint and formatting checks pass; **423 files**
  are formatted. No Taichi kernel file was formatted or modified in this tranche.
- [x] Generated bindings check passes for four modules/five kernels. All ten
  changed Python files parse, and `git diff --check` passes.
- [ ] The final canonical fast suite is **not fully green**: **607 passed,
  1 failed, 3959 deselected**. The sole failure is the container MathTex/dvisvgm
  SVG-grouping baseline mismatch (maximum deviation 221 at frame 4). A fresh
  unchanged-parent run produced **607 passed, the same failure**. The exact
  output comparison above was repeated after the final fast rerun.
- [ ] Full `pytest -q` was attempted and reached its **600-second limit
  (exit 124)** without a final summary. Three failures appeared before timeout:
  `text_and_math.rst:115`, `text_and_math.rst:190` and `mob_gallery.rst:329` in
  `test_doc_examples.py`. Each was separately reproduced on unchanged `077d2660`
  and on the new code, with missing MathTex SVG groups and indexing errors.
  No full-suite pass or completed heavy-render comparison is claimed.
- [ ] CUDA, Metal and AMD runtime validation, alternating warm performance and
  actual total-device peak-memory measurements remain outstanding.

The PR remains draft. No rendering baselines or fast-suite membership were
changed. CPU tests used the supplied editable-install environment with daemon
handoff disabled; the unchanged-parent checkout and compaction probes verified
their source paths. Focused, fast and probe results overlap and are not an
aggregate unique-test count. Earlier intermediate focused runs are superseded
by the final 22-module result. Previous-tranche route fixtures and validation
below are historical, not newly rerun GPU or path-tracer coverage.

### Third-tranche validation (historical; unchanged below)

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

## Recommended next work

1. Validate actual CUDA/Metal/AMD execution: full-rank grouping, exact integer
   copies, caller destinations, tiny floors and the new staged floating/reference
   work. Measure alternating warm time and real total-device peak memory; complete
   broader/heavy-render validation before declaring the PR ready.
2. Use those measurements to decide whether further lifetime reductions and
   remaining reference prefix/index-conversion work are worthwhile. Library
   unique/sort/scan/nonzero internals still allocate independently of result
   ownership; do not replace them merely to label every tensor arena-owned.
3. Benchmark indexed shadow tracing and batched endpoint downloads before changing
   payload gathering or full host-offset caching. Independent integer-scan and
   broader named-context cleanup remain separate tasks.

## Eighth-tranche publication record (historical)

The remote implementation branch was rechecked at exact base
`a222eca1468990b459528641c3a336a23b888f51` before publication. Direct container
network push still could not resolve `github.com`, so the documented one-off
Actions transport was used. The successful publisher run `34595612033` verified
the staged XZ and patch SHA-256 values, reapplied all four commits with their
original metadata, required final implementation head
`25ab7967cf2fe2befc8c460b68c0526b0561f83d`, required tree
`c64cac9239c9ccf41e9ff90bfc117d6fbbc82f5c`, checked generated arena bindings and
performed a normal fast-forward push. Independent GitHub comparison reports exactly
four implementation commits and no transport files in the target branch.

The transport branch could not be deleted under the repository branch rule, so it
was cleaned with a normal fast-forward commit restoring the exact seventh-tranche
source tree; comparison against `a222eca1` reports no changed files. No force push,
master write or helper ancestry entered the implementation branch. This publication
record is transport verification; the runtime validation results above remain the
ones performed before publication.
