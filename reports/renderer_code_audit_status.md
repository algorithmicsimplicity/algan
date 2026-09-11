# Deterministic renderer code audit — implementation status

Updated: September 11, 2026. Branch: `codex/renderer-audit-memory-cleanup`.
Pull request: #130, targeting `master` (not merged).

The original audit examined `f2073d718364617b35ed86028efefcd0ccda5606`.
The first implementation is `0b20f817ac25e4fe0aba9f9816b8485c9f26e8ec`.
The second implementation is `60597722f0e472ede8a173251f54282694172371`.
The third implementation is `077d2660c585e2f73b727dfd82f371a828c595bd`.
The fourth implementation is `dd4bfadc52387ac6249878f2602d094d8349862e`.
The fifth implementation is `234ac00663aa54a1837c4a2d8441fe52a224697c`.
This document accompanies the sixth implementation, directly on that fifth
commit. Checked items describe code present on this branch, not a promise that
all platforms or the full test suite have passed.

Legend: `[x]` implemented; `[ ]` remaining. A **Partial** heading means the
listed completed sub-items are present but the entire audit recommendation is
not finished. “Unchanged” means a contract was preserved, not newly implemented.

## Summary of this update

The sixth tranche gives sorted pixel/depth/coverage/mask payloads a named
caller-owned destination and moves their production storage into compaction's
forward workspace. Rank/class grouping accepts an exact caller-owned inverse;
native rank grouping writes it directly, while PyTorch unique retains its
internal temporary inverse and copies it into the destination. Dynamic group
labels are still allocated by the grouping operation, then adopted into the
production result stage. This is not a claim that unique's workspace vanished.

Class grouping now has one implementation shared with the compatibility entry
point. Packed-key builders widen int32 inputs before multiplication, rather
than relying on an int64 destination to widen the operation. Its pair-sort path stages keys, permutation, boundaries and IDs without
constructing a wide MPS key. A checked boundary-to-group-ID scan replaces the
remaining repeated boolean-cast/scan/subtract expressions in sheet compaction.
Final gathers use explicit destinations; the ordered band map is reused by
sibling weighting and sample-depth classification, and an unused representative
index gather is skipped. Mask flags and band-membership scratch are staged.

An outer compaction workspace scope protects these new forward allocations on
success and failure. Persistent sheet outputs still live in reverse storage;
standalone diagnostics retain ordinary ownership. This bounds the new result
storage by the entire compaction call, not each individual last consumer. It
does not establish a reduction in total peak memory or warm time. Preprocessing
metadata, rank-pooling maps and several expressions remain external. The
conflict-rank ceiling, rendering mathematics and kernel ABI are unchanged.

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
- [ ] Move remaining preprocessing metadata, rank-pooling maps and tensor expressions into explicit destinations or shorter stages. PyTorch unique still creates temporary inverse/key arrays; avoid describing their copies as elimination of library workspace.
- [ ] Evaluate broader fused finalization only after preserving the current accumulation/rounding boundaries.

`sheet_buffers.py` and `sheet_output_taichi.py` own persistent resolver output.
`sheet_workspace.py` owns staged scratch and caller result lifetimes;
`sheet_reduction_buffers.py` and `sheet_statistics.py` name result destinations.
`sheet_order.py` owns stable-order composition scratch; `device_sort.py`
accepts native destinations. `array_ops.gather_rows` and `array_copy_taichi.py`
provide exact row-copy destinations. `sheet_fragments.py` names sorted payloads;
`sheet_grouping.py` owns shared grouping and inverse destinations. Preprocessing
metadata, rank-pooling maps, dynamic unique temporaries, remaining expressions
and library sort/scan workspace still require external storage. The workspace counter is not a total-device peak-memory measurement.

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

The retained-floor mechanism still protects genuine batch tables and trees;
this is not a redesign of `ManualMemory` as an arbitrary region allocator.
Current batch publishers run before coverage or at the clean restart boundary,
so they no longer pin coverage/tile allocations. Tests deliberately force a
false-positive deferral eligibility decision; ordinary reflective scenes are
not claimed to select deferral. That exceptional path pays one chunk replay,
not an eager BVH reservation for every otherwise split-free batch.

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
- [x] Document shared pixel-run invalidation and caller-owned result lifetimes, and correct earlier present-tense descriptions of external reduction output storage.
- [x] Correct deferred-publication/retry comments and document clean-boundary restarts, exact gathers and stable-sort output lifetimes.
- [x] Consolidate the compatibility class-group implementation and correct touched comments about external reduction/sorted-payload ownership.
- [ ] Continue removing obsolete comments and remaining unsupported-path compatibility plumbing only after proving the callers and diagnostic fixtures no longer require it.

## Validation

### Sixth-tranche validation in this container

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

## Recommended next implementation order

1. Continue ownership propagation through preprocessing metadata, rank-pooling
   maps and remaining tensor expressions. Shorten result lifetimes where feasible;
   the new sorted-payload scope deliberately lasts through compaction, and
   PyTorch unique still allocates its intermediate inverse/key arrays.
2. Validate actual CUDA/Metal/AMD execution, including the native radix and
   integer-copy destinations, and measure alternating warm time and real peak
   memory before making performance claims.
3. Benchmark indexed shadow tracing and batched endpoint downloads before
   replacing the current gathering and host-offset-cache policies.
4. Treat the conflict-rank ceiling as a separate semantic/capacity change with
   deep same-surface transparency fixtures, not part of a copy refactor.
