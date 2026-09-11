# Renderer memory ownership: audit implementation

These changes implement the September 11, 2026 code audit against master
`f2073d718364617b35ed86028efefcd0ccda5606`. It changes ownership, copies and
validation, not analytic coverage, material transport, depth ordering or the
same-surface conflict-rank limit.

## Allocation and binding contracts

`ManualMemory.get_tensor` validates integer, nonnegative extents before pointer
arithmetic. It constructs the typed view and applies optional poisoning before
committing either pointer, the high-water mark or the allocation record. Scalars
and zero-sized arrays are supported in both allocation directions. Alignment is
still charged even when an empty allocation needs to align a pointer.
`clone` and `cast` use `copy_`, which also supports scalar destinations.

The argument-table cache includes tensor strides in its key. A contiguous square
array cannot warm the cache for its noncontiguous transpose. Cache keys still
contain no tensors, and whole-arena views are still rebuilt per launch rather
than retained by the cache. The first tranche left kernel argument layouts unchanged. The second tranche
adds typed metadata through the generated layout described below.

## Raw fragments versus resolver records

`prepare_sparse_raster_coverage(..., retain_fragments=True)` keeps the diagnostic
contract: raw `frag_*` arrays and `run_offsets` coexist with the sheets. Normal
tracer calls explicitly pass False unless fragment capture was armed at the
start of discovery. With False, the raw records live in forward discovery
scratch, and only covered-pixel indices and resolver-ready sheet records live
at the reverse end after the function returns. Raw keys are omitted from the
returned dictionary rather than exposing reclaimed tensor views.

This releases 32 bytes per retained raw fragment plus the raw CSR after
compaction, subject to alignment. It does **not** remove the need for raw
fragments during discovery or claim the same reduction in peak memory. The
existing discovery-footprint estimate still includes their coexisting storage.
The allocation order, not Python reference deletion, effects reclamation.

Capture requests are sampled once at the ownership decision. A request armed
while discovery is already running waits for a later chunk; disarming still
prevents capture inside `fragment_capture.capture`. Diagnostic callers of
`prepare_sparse_raster_coverage` retain raw records by default.

`compact_sheets(..., diagnostics=True)` also preserves its standalone diagnostic
API. Production passes False, omitting `sheet_nfrag`, `sheet_fused`, `num_groups`
and `num_split_groups`. The separate group-statistics scan and final diagnostic
gathers do not run. The shared native reductions specialize away unused fused
lane-duplicate and fragment-count stores; their required union, area and
reference statistics remain enabled. Truncation reporting remains unconditional. Final key gathering composes nearest-fragment indices
first, using exact integer gathering on MPS, then gathers the packed key once.

## Integer count/scan/write

`array_ops.csr_offsets` writes an integer `n + 1` CSR array, including the total.
It accepts an explicit arena-backed output, checks shape/dtype/device/layout and
rejects overlap before writing. Callers guarantee nonnegative counts and a
representable total; the helper performs no value-validation readback.

Raster discovery allocates one count array and one acceptance-mask array and
passes disjoint slices to each geometry/class count kernel. One wide prefix
scan replaces concatenation, a full-array conversion, exclusive-prefix
subtraction and a separate sum. One boundary readback supplies each write
pass's range and the total. Candidate and fragment totals are checked before
narrowing to int32 kernel indexing. The raw-fragment CSR reuses the pixel-run scan described below; diagnostic
retention copies its offsets into reverse storage without another scan.
Candidate expansion and native conflict-rank group
counts use the same terminal-offset contract. Float shell-coverage prefixes
retain their accumulation dtype and global scan/subtraction boundaries; they
are not replaced by this integer helper. PyTorch may still allocate internal
scan workspace.

## Shared shadow copies

Both primary-sheet and deferred-wavefront shadow submission use
`shadow_queue._gather_shadow_payload`. One kernel gathers position, smooth and
face normals, frame, mask and enabled footprint/terminator fields into arena
storage. Disabled fields alias their existing placeholders and are never read
by the corresponding specialized kernel arm. A named tuple records the payload
layout. The enclosing tile/iteration temporary scope owns the gathered arrays
until tracing completes.

A frozen `ShadowTraceContext` holds the scene/light launch context and owns one
explicit trace launch. It receives the currently selected triangle and Bezier
BVHs on every call, so a late build or an opaque-tree choice is not hidden by a
cached placeholder. The concrete triangle tree determines the refit template.
Source identity, emitter sampling, footprint generation, sorting policy and
geometry-presence gates remain explicit caller decisions. Sharing the launch
does not merge those policies. Deferred visibility is scattered
directly into the already-initialized padded table; absent events and unused
light slots remain one, without an intermediate all-lit `filled` tensor. These
copies retain integer values as integers. MPS launch indices go through
`kernel_index`, as in the existing renderer kernels.

## Memory-failure retries

The sparse primary resolve and bounce drain share `_shrink_sparse_memory_retry`.
After a memory failure both halve the primary count and shrink a splitting pool.
Previously the drain kept the pool fixed, so reducing primary work could leave
almost all arena storage occupied and repeatedly fail to allocate scratch. Pool
**overflow** retries still keep their pool: inheriting its spare slots is correct
for overflow, but not for allocation exhaustion. Injected failures at each stage
verify the next pool is smaller and the resulting frame matches the unfailed
render within one byte-channel value.

## Validation and next work

Regression tests cover cache-layout collision, scalar/empty/invalid arena
allocations, failure atomicity, integer scan destinations, diagnostic-free sheet
parity, exact shadow payloads and visibility padding. A real-render test compares
retained and temporary raw-fragment routes, measures their retained allocation
difference, overwrites freed forward scratch and checks every returned sheet
record before resolving one- and two-frame mixed-geometry scenes. The existing
viewer tests continue to exercise armed fragment capture.

CPU validation does not establish CUDA/Metal/AMD parity or a performance gain.
Use the repository's GPU harnesses for those checks before drawing performance
conclusions. No rendering baseline should be regenerated to hide a mismatch.

Remaining audit work includes sheet-compaction sorting/grouping intermediates
and tensor expressions. The audited reverse-storage retention hole is closed
by the fifth-tranche chunk restart described below. Pixel-run CSR sharing and
explicit invalidation are implemented in the fourth tranche.
Removing the conflict-rank ceiling is a separate behavior change. The typed
metadata, generated ABI, prepared-batch policy and structured attempt cleanup
are implemented in the tranches described below.


## Second tranche: typed render metadata

`render_metadata.py` owns two fixed layouts. Float32 carries the layer-order
scalar, environment intensity and far clip. Int32 carries the environment texel
offset and dimensions, maximum bounce count and per-tile glossy accumulator
base. The sheet and wavefront shade kernels bind both through their existing
float/int arenas. Address/count fields never pass through float32 or add a
floating rounding bias; metadata is initialized on every route rather than
being inferred from optional array lengths. Host integer validation precedes
allocation. The glossy base remains a mutable per-tile integer; it is not a
new specialization key or a reason to rebuild the batch tables.

## Second tranche: generated arena ABI

`arena_layouts.py` is the source of truth for the original positional arguments
and each bound array's dtype/rank in all five arena-packed kernels. A plain
string denotes a kept argument; a triple denotes a bound array. The standalone
`python scripts/generate_arena_bindings.py` tool reads that literal schema
without importing Algan or initializing a compiler. It generates static
prologues, specs and parameter lists inside the existing four kernel modules.

Ordinary kernel signatures retain their annotations and explanatory comments.
The generator validates them before writing any file. Existing prologue,
wrapper-signature, Metal buffer-budget and explicit launch-arity tests remain
in force; a separate regeneration test detects stale source. Run with `--check`
for a no-write validation. There is no runtime code generation, no change to
the established hot/cold argument split, and no replacement of checked launch
sites by unchecked splats.

## Second tranche: direct sheet outputs and scoped sort destinations

`compact_sheets(..., diagnostics=False, resolver_memory=memory)` returns named
`SheetBuffers` directly in reverse-arena storage. The final copy kernel follows
`nearest[final[i]]` for the depth key and `representative[final[i]]` for shading
reference/barycentrics/cap. Coverage and masks are already-finalized resolver
weights, copied bit-for-bit without changing accumulation or sibling arithmetic.
A lower-bound kernel builds the int32 CSR directly in its destination. This
removes allocator-owned final key/reference/barycentric/cap payloads and the
caller's subsequent copy of the final record.

Standalone compaction keeps its diagnostic dictionary and raw-area semantics;
it remains the reference arm in tests. Supplying `resolver_memory` together
with `diagnostics=True` is rejected rather than silently returning another
representation. Native per-pixel and final-walk sort permutations also write
into forward arena scratch. The caller's discovery scope still owns the surrounding raw records. The sixth
tranche additionally wraps compaction's own workspace in an exception-safe
forward scope. Both permutations are included conservatively in the discovery
footprint estimate, even where a native sort falls back to PyTorch. The forward
scope can be overwritten after return without damaging persistent sheet data.

This is **not** a complete arena conversion of compaction. The third tranche
adds short stage lifetimes for many reductions and temporary tables. The fourth
tranche moves band-composite, final band-reduction, reference-selection and
sibling-weight results into caller-owned stages. The fifth and sixth tranches
extend ownership to sorting, sorted payloads and grouping inverses. The seventh
adds preprocessing, rank-pooling maps and sample-depth metadata as described
below. Remaining expressions and library workspace still need external
headroom. The standalone sheet CSR uses lower
bounds independently of `sheet_metadata_kernel`, which controls diagnostic
counting only; it no longer selects two unrelated algorithms together.

## Second tranche: shared bounds and ray-state ownership

Triangles and circuits share `_pack_screen_bounds`; projection, straddler
clipping and opacity-uncertainty classification remain at their original sites.
The common finishing step retains inclusive margins, unbounded front-facing
fallbacks, valid/translucent classification and persistent allocation direction.
`triangle_scene_bounds` caches extrema and the original tensor-precision norm
once per immutable merged scene. Shadow epsilon and Morton quantization derive
from that same record, with their existing degenerate/empty fallbacks. The norm
is deliberately not recomputed with host double-precision arithmetic.

`RayState` names the existing eleven-element host tuple without changing its
layout or placeholder aliases. The sparse accumulator index in integer column
4 remains live. The classic tile loop now wraps allocation, drain, allocator
readback and compositing in a `memory.temp(clear_persist=True)` attempt scope,
so all exits restore both arena ends. The third tranche applies equivalent structured ownership to sparse attempts,
with the additional retained-BVH contract described below. The retained floor
itself remains necessary until lifetime regions are separated further.

The complete, item-by-item branch checklist and current validation record are
in `reports/renderer_code_audit_status.md` at the repository root.


## Third tranche: staged compaction scratch

`CompactionWorkspace` provides nested `stage()` scopes. Arena-backed scratch is
allocated at the forward end and rewound on every exit, including exceptions;
reverse-end resolver records are unaffected. Standalone diagnostic callers use
the same helper with an ordinary-allocator fallback. The workspace retains no
tensor references. Its `copy` always makes a distinct copy, even when source and
destination dtypes match: shell-prefix scratch must not alias fragment coverage
in the float32/MPS-friendly configuration.

The converted scratch includes band accumulation, conflict-rank lane scans,
rank-group counts/CSR, shell-ceiling reordered coverage and prefixes, band
reference statistics, sibling membership/count arithmetic, lane first-owner
and depth tables, and final lost-lane masks. Nested stages reuse storage after
each consumer rather than retaining all arrays until compaction ends. A named
`SheetStatistics` record names the reduction results that cross stage boundaries.
They were ordinary allocations in this tranche; the fourth tranche adds explicit
output destinations. Explicit rank and lane-depth output destinations are checked
for layout and conservative byte-range overlap before writing.

Float64 accumulation still rounds only after its completed reduction; the
float32 compatibility arm retains float32 accumulation. Shell-ceiling scans
keep the old global prefix and exclusive-prefix subtraction boundaries. These
changes do not fuse floating-point operations across earlier rounding points.
The fixed rank clamp and its reporting are unchanged.

`peak_bytes` is the maximum overlap of allocations made through that workspace,
including alignment. Discovery adds it to the existing raw/final-record and
permutation estimate. It does not sum disjoint stages or claim to measure total
PyTorch, compiler, driver or device peak memory.

## Third tranche: one prepared-batch execution policy

`resolve_batch_policy` reads live settings and immutable scene facts once for a
prepared batch. Frozen, tensor-free `BatchExecutionPolicy` and `WavefrontPolicy`
records carry the primary route and fallback reasons, sample count, requested
and effective AA, in-place versus supersampled frame scale, shadow capability,
fragment/custom-scatter requirements, continuation/IOR state width, pool ratio,
and core wavefront specialization decisions.

Preflight attaches this policy to the prepared primitive batch. Frame sizing,
capability validation, device upload dispatch and wavefront execution reuse it.
The render loop clears it with the corresponding prepared/device scene; a new
render resolves current settings again. In-place AA charges an output-resolution
frame, rather than a supersampled frame that it never allocates. The render plan
exposes route, effective AA and fallback reasons. Standalone low-level calls
still resolve their own policy when none is supplied.

BVH objects are not part of this policy: late publication can replace them.
Defensive data/route consistency checks remain, and independent microkernel
optimization gates still have their own readers. Concurrent mutation of settings
inside a running batch is not a supported contract.

## Third tranche: sparse attempts and deferred BVH publication

Each sparse tile attempt scopes allocation, resolve, drain, allocator readback,
and final compositing with `memory.temp(clear_persist=True, persist_floor=...)`.
All retry, success, break and exceptional exits restore forward scratch and
reverse temporary storage while retaining the already-published batch floor.
The fifth tranche moves late publication outside the chunk's temporary scopes.
Memory-failure retries still shrink both primary work and pool size;
capacity-overflow retries retain their separate pool policy.

Construction and arena publication are distinct states. If BVH construction
succeeds but its arena copy fails, `bvh_deferred` is already false.
`bvh_rehome_pending` keeps the publication obligation visible to the next retry.
After successful copying, local tree references are rebound and the retained
reverse floor is published. Copy failure leaves the constructed source trees
available and the pending flag set; partial arena scratch is reclaimed by the
publication transaction and enclosing scopes.

Tests force a deferred eligibility false positive, then inject failures after
construction, inside arena copying, and during resolve/drain/readback/compositing.
They check retry output and poison reclaimed storage while verifying retained
BVH bytes. These tests originally proved safe cleanup subject to a retained
floor. The fifth tranche additionally proves publication occurs at the clean
batch/chunk boundary rather than retaining intervening reverse allocations.


## Fourth tranche: shared pixel-run CSR

`PixelRunCSR` describes one immutable, pixel-sorted fragment stream: covered
pixels, counts, terminal offsets and the fragment count. Discovery builds it
after sorting. Opaque-prefix truncation and one-mesh classification use the
same starts/counts, regardless of their independent kernel gates. The final raw
fragment record reuses its offsets instead of scanning the counts again.

The owner discards the record before filtering the stream and builds a fresh
record for the filtered rows. Changes to coverage or mask flags do not change
run boundaries. The count-identity/count-length guard catches accidentally
mixing two records without reading device values; it is not a content hash and
cannot detect arbitrary in-place mutation. Callers must also replace the record
after any reordering or membership change, including changes that keep the same
number of rows. No long-lived tensor cache was added.

Counts and covered-pixel tensors remain ordinary allocations. Production offsets
use forward discovery storage and signed int32, after checking that the fragment
count fits. Standalone construction retains int64 offsets. Diagnostic/capture
retention copies those already-computed offsets into persistent reverse storage;
normal rendering keeps them only until compaction finishes. Both the initial and
replacement CSR can occupy forward storage until discovery returns. The budget
charges their actual allocation deltas, including alignment; the existing raw
CSR allowance remains conservative and also covers a diagnostic retained copy.

## Fourth tranche: caller-owned reduction results

`BandComposite`, `BandReduction`, `SheetStatistics` and `SheetWeights` distinguish
results from each helper's scratch. Their `allocate` methods use the caller's
already-open `CompactionWorkspace.stage()`. The helper opens a nested scratch
stage, writes the supplied `out` buffers and rewinds only its own temporaries.
Layout and byte-range overlap checks precede all writes to supplied outputs.
The helpers retain ordinary-allocation defaults for standalone/reference use;
empty and singleton sibling outputs are copied when a destination is supplied
rather than silently aliasing their inputs.

Production compaction has three nested result lifetimes. The outer stage owns
band-composite area, union, correction and split flags through shade-class and
rank grouping. The next stage owns final band area/mask and nearest/dominant
reference statistics through final ordering. The innermost stage owns final
sibling weights, masks and lane-depth work until `finish_sheet_buffers` copies
the resolver records into reverse storage. Returning or raising unwinds every
forward result stage. Standalone diagnostic outputs remain ordinary owned
tensors and do not escape as dangling arena views. Rank-pooling area/union
results have a shorter scope ending before the next grouping operation.

Area accumulation keeps its float64-to-float32 boundary, and the float32
compatibility path still accumulates in float32. Narrow integer reference
reductions use int32 scratch then copy to exact int64 result buffers. The final
coverage expressions retain their original rounding and signed-continuation
conventions. There is no additional kernel binding, new specialization gate,
reordered depth walk or change to the rank ceiling.

The workspace high-water counter includes these overlapping output stages.
This is a change in allocation ownership, not a measured reduction in total
peak memory. Gathered streams, grouping, remaining tensor expressions and native
library/compiler workspace still require external headroom. The fifth tranche
adds sort and gather workspace ownership below.


## Fifth tranche: stable sort and exact gather destinations

`sheet_order.stable_lexsort` accepts a caller-owned int64 permutation and a
`CompactionWorkspace`. Its PyTorch fallback keeps one composed permutation and
one pass-index buffer, then stages each gathered key and sort-value buffer.
Their storage is reused between key passes. The result remains valid after the
scratch stage closes, including when a standalone caller supplies a workspace
but requests an ordinary allocated result. Stable least-significant-first
composition, float NaN/signed-zero behavior, and integer key precision are
unchanged.

`device_sort.stable_argsort` and `stable_lexsort` accept int32 destinations and
stage their radix key, permutation, count and scan scratch. The original GPU,
compiler, dtype, size and opt-in gates remain. No CPU block-radix implementation
was added. The sheet wrapper widens the native permutation directly into the
caller-owned int64 output. Tests with a CPU oracle check native launch ownership
and composition; they do not constitute GPU radix-sort runtime validation.

Per-pixel ordering and final-walk ordering allocate their outputs directly in
forward discovery storage on every arm, not only the run-local kernel arm.
Shell key/depth ordering and its gathered run keys use the surrounding shell
stage. The validated packed CUDA path keeps its eligibility checks, full depth
bits, signed-int64 capacity check and PyTorch stable-sort choice; its packed key,
per-column deltas and sort values now accept staged workspace. Delta storage is
released between columns. Standalone calls without destinations keep their
existing allocation defaults. Discovery charges the two long-lived permutation
arrays separately and includes overlapping sort scratch in `peak_bytes`.

`array_ops.gather_rows` copies dimension-zero rows into a contiguous destination
without changing dtype. It checks index metadata, output metadata and byte-range
overlap before writes; valid nonnegative index values remain the caller's
responsibility. Ordinary CPU/CUDA and floating gathers use `index_select(out=)`.
The MPS-friendly integer route uses a local exact-copy kernel, with the existing
advanced-indexing fallback when no local kernel is available. That fallback may
still allocate. `CompactionWorkspace.gather` owns a result until its stage ends.
The rank-pooling per-fragment gather is released immediately after the reduction,
before constructing the pooling key. No float-to-integer representation shortcut
was introduced.

## Fifth tranche: late-BVH publication at a clean chunk boundary

A sparse resolve can discover a continuation after merge-time BVH deferral.
Previously it built and retained the trees underneath already-allocated reverse
coverage records, so the retained floor also kept the intervening records alive
for later chunks. Moving pointers without moving the trees would corrupt them;
reserving a worst-case tree region eagerly would defeat deferral's memory saving.

The tracer now raises a private restart signal at that discovery point. Tile
and coverage scopes unwind first; `render_chunk` rewinds its output and other
forward state, preserving only previously published batch tables. It restores
the chunk-entry truncation and path-sample counters, clears the discarded
traceback frames, and publishes the trees transactionally at that clean boundary.
It then restarts the same chunk, including background prefill. This is a bounded
restart: successful publication clears the deferred/pending flags, and later
chunks reuse the trees. No partial composite or discarded statistic is accepted.

A publication failure restores both arena ends. The constructed external trees
and `bvh_rehome_pending` survive a failed copy, so one allocator-reclaim retry
can finish publication without rebuilding. If that clean-boundary retry still
cannot fit, it raises `OutOfRenderMemory` for prepared-batch recovery; halving a
ray tile cannot free more storage there. Non-memory exceptions are not retried.
Early shadow, classic and path-tracer builds use the same transactional publisher
before coverage allocation. The retained-floor mechanism still protects real
batch tables and trees across chunks; it no longer pins chunk-local coverage
beneath a late-built tree.

Real-render regression fixtures deliberately force a false-positive deferral
eligibility decision, assert the publication pointers are at the batch floor,
poison a discarded background, inject a later chunk split, and compare output to
eager construction. They assert one build, clean statistics, and reuse across
the split chunks. Separate tests inject partial forward/reverse publication
allocations and non-memory failures and verify bounded retries and persistent
sentinels. These fixtures do not claim ordinary reflective scenes are normally
eligible for deferral, or establish a performance or total-device-memory result.


## Sixth tranche: sorted payloads, grouping inverses and final gathers

`SortedFragments` groups the sorted pixel IDs, exact depth values, private
coverage values and mask words. `gather_sorted_fragments` validates every input
and destination's metadata and disjointness before writing any field, and uses
`gather_rows` rather than converting integer values through floating point.
Private coverage ownership is required because the closed-shell ceiling changes
it in place. Production fields occupy forward workspace; diagnostic outputs
remain independently allocated. No per-fragment Python objects are introduced.

`RankGroups` names inverse IDs and per-group parent/rank descriptors. Rank and
class grouping accept an exact, contiguous int64 inverse destination. Native
rank grouping writes it directly; source integer widths are normalized in
scratch for the count reduction. The torch unique path still uses its own
inverse temporarily, then copies it into the supplied destination. Dynamic
labels retain ordinary ownership on return from a grouping helper; production
adopts them into its surrounding stage after the helper's scratch has unwound.
The uniform-class fast path honors an output destination rather than returning
an input alias. Input values retain their existing domain requirements: dense
ordered rank parents, clamped ranks, and class values below their packing base.

`sheet_grouping.class_groups` is the one implementation shared by compaction
and the compatibility spelling in `mps_compat`. Non-MPS grouping preserves its
sorted packed-key unique policy. The MPS-friendly arm preserves stable pair
ordering with known-int32 band/class keys, never a wide composite key. Its sort
keys, permutation, comparisons and boundary scan are staged; its result inverse
outlives them. Metadata and overlap rejection precedes writes. A valid output
is not promised to be transactionally unchanged after an execution failure;
workspace pointers are restored so the caller can discard and retry the result.

`group_ids_from_starts` implements the existing boolean boundary scan as one
inclusive integer cumsum into its destination followed by an in-place subtract.
It avoids the materialized boolean-to-integer input and subtraction result.
Both int32 and int64 outputs are supported. Metadata, byte ranges and the
inclusive scan's worst-case capacity are checked without reading device values.
Callers supply a true first flag for a normal nonempty stream; a false first
flag retains the old expression's -1 result rather than silently changing it.
This helper replaces integer grouping scans, not floating shell prefixes.

Final sheet coverage, masks, pixels and sample-depth tables use exact destination
gathers. Sibling weights and sample-depth classification share the final band
map. Persistent finalization already follows the representative indices itself,
so compaction skips its extra representative gather unless sample-depth work
needs it. Mask flag construction and membership-count scratch use short nested
stages; all floating coverage arithmetic and output ordering remain unchanged.

An outer `CompactionWorkspace.stage()` now bounds the new sorted-payload and
group-result allocations. With the normal shared workspace/resolver arena, it
also reclaims direct sort permutations on return or exception. A separately
supplied workspace only owns its own arena: callers must still scope direct
resolver-arena permutations when using different arenas. Existing discovery
scopes remain in place. Final reverse-arena output survives; diagnostics contain
no references to reclaimed workspace. Individual result arrays are not freed at
their last Python `del`: they remain until their owning stage closes.

The workspace's overlapping-byte counter automatically accounts for the new
fields and adopted labels, including alignment. Discovery still conservatively
charges direct sort permutations separately. At this tranche, preprocessing
metadata and rank-pooling maps still needed external storage; the seventh-tranche
changes below supersede that ownership description. PyTorch unique intermediates
and compiler/library/driver workspace continue to need external headroom. No total-device peak reduction or speedup follows merely from
moving the result arrays into the arena. The rank ceiling and all kernel layouts
are unchanged.


## Seventh tranche: preprocessing, rank pooling and sample-depth metadata

`sheet_preprocessing.FragmentMetadata` gives decoded pixels, exact float32 depth
bits, chunk-relative frames, triangle flags, safe references and surface/facing
keys checked caller-owned destinations. All field metadata and pairwise/input
byte ranges are validated before any destination is written. Surface IDs are
widened before multiplication; circuit fragments retain distinct negative keys.
`array_ops.gather_frame_table` shares the wrapped frame/primitive lookup with an
explicit destination and a short-lived int64 flattened index. It uses the exact
row-copy policy already used by sorting, including the local MPS integer-copy
arm and its allocating fallback. A strided table may still require a temporary
flattening copy. Valid primitive indices remain the producer's responsibility;
no device readback was added to check bounds.

Compaction allocates the sorted payload, order and metadata consumers before a
nested preprocessing stage. That stage owns decoded keys and reference maps.
Raw shading classes are built only after sorting and released immediately after
their gather; the per-frame class table is nested inside that lifetime. Group
comparisons and primitive-split results have their own shorter stages. The
primitive-depth-slope table is staged too, but its per-block floating-point
expressions retain their original arithmetic and ordinary temporary ownership.
The preprocessing region closes before conflict-rank grouping. A regression test
checks that the rank inverse actually reuses the decoded pixel buffer's address,
not just that a later Python reference was deleted. Sorted payloads, group flags
and shared positions still last through their broader owning compaction stage.

`RankPoolGroups` names the compositing-group count and optional inverse. A checked
int64 output can own the inverse; `None` still means that no pooling took place,
and a supplied but unused destination has unspecified contents. The parent map,
full-union/area flags, per-fragment map and key are staged. The key survives a
nested reduction stage, so area/union storage and membership flags are released
before the second unique operation. Production uses explicit destinations for
pooled/class maps and releases class-composition scratch before final reductions.
PyTorch unique still allocates its own inverse before copying it to the caller;
this is ownership propagation, not elimination of library workspace.

`SampleDepthMetadata` names final sample masks, surface IDs and enforcer/subject
flags. It preserves the float32 `(coverage - 1).abs()` dust check, nonnegative
weight gate (including negative zero), full-mask rule, circuit exclusion and
multi-sheet-band exemption. Lookup indices and intermediate predicates live in
short scratch stages; all classification and depth-competition data is reclaimed
after lose bits are applied. Production ORs those bits into existing mask
storage; exact mask aliases are safe because OR is idempotent. Standalone output
allocation behavior is retained. No coverage sum, shell prefix, threshold,
conflict-rank ceiling or native kernel signature changed.

The workspace counter includes these staged allocations and their alignment.
Discovery also charges worst-case alignment for its two directly allocated
int64 sort permutations. These estimates exclude compiler/library/driver
storage and cannot establish total-device peak savings or faster warm renders.
At the seventh tranche, closed-shell eligibility still created dynamic ordinary-owned
arrays before production adopted them. The eighth-tranche stages below replace
those arrays; several block floating expressions remain unstaged.
Runtime failures restore scratch pointers, not partially written caller results;
callers discard those results on failure. Persistent reverse outputs and prior
arena sentinels remain protected by the existing discovery/compaction scopes.


## Eighth tranche: closed-shell metadata and coverage stages

`sheet_shells.ShellSegments` names the sorted shell segment key and facing flag.
`shell_segments` accepts checked caller destinations or returns ordinary-owned
standalone results. Closed declarations and surface IDs use the exact shared
frame/primitive gather, including independent animated-table wrapping and the
existing circuit-safe reference policy. The stride still uses the maximum of
all looked-up surface IDs, not just active closed triangles: the order of the
subsequent global floating-point scan must not change. Empty or inactive inputs
leave supplied outputs untouched and skip the surface lookup.

`apply_shell_ceiling` owns both the native and reference arms' temporary arrays.
Both retain the original global exclusive coverage prefix and the separate
within-segment subtraction. The reference arm names group IDs, first positions,
spent coverage, face sums and scale destinations, reclaims its grouping and face
scratch in nested stages, and rounds the maximum face sum through float32 before
converting back to the accumulator dtype. The denominator floor also changes the
final coverage multiplicand, as before. Float32 scratch is always a private copy,
not an alias of the mutable coverage. Strided diagnostic sort inputs get staged
contiguous copies; normal contiguous inputs are borrowed.

The shell ceiling now runs at the end of fragment preprocessing, before conflict
rank grouping. It mutates only private sorted coverage; conflict ranks still read
the unchanged sample masks. All shell lookups, result keys and reduction scratch
therefore end before the rank-group stage. Regression tests inspect both active
stage depth and forward pointer at that boundary, poison reclaimed storage, and
verify resolver outputs. Failures during lookup, sorting, scan, face reduction,
copy and native application unwind scratch. As with the other helpers, a runtime
failure can leave a caller result partially written; the enclosing attempt must
discard it, rather than expecting a transactional output buffer.

Declared shells still spend `max(front_area, back_area)` in true depth order,
without clamping that allowance to one. Same-facing self-overlap, undeclared or
transmissive surfaces, and circuit pass-through behavior are unchanged. No kernel
signature, packed ABI, accumulation order, conflict-rank capacity, or material
policy is changed by this ownership refactor. Library sort/scan/nonzero workspace
and optional integer-index conversions still need external headroom. Workspace
accounting is not a measurement of total device peak memory or warm time.
