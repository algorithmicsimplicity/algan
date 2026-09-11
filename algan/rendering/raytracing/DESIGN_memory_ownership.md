# Renderer memory ownership: first audit implementation

This change starts the September 11, 2026 code audit against master
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
than retained by the cache. Kernel argument layouts are unchanged.

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
gathers do not run. Outputs of shared reductions that also support rendering
are intentionally not redesigned in this first change. Truncation reporting
remains unconditional. Final key gathering composes nearest-fragment indices
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
narrowing to int32 kernel indexing. The raw-fragment CSR is also written directly
into its arena destination. Float prefix sums used by shell coverage are not
changed. PyTorch may still allocate internal scan workspace.

## Shared shadow copies

Both primary-sheet and deferred-wavefront shadow submission use
`shadow_queue._gather_shadow_payload`. One kernel gathers position, smooth and
face normals, frame, mask and enabled footprint/terminator fields into arena
storage. Disabled fields alias their existing placeholders and are never read
by the corresponding specialized kernel arm. A named tuple records the payload
layout. The enclosing tile/iteration temporary scope owns the gathered arrays
until tracing completes.

Source identity, emitter sampling, footprint generation, sorting policy and the
explicit trace launch arguments remain at their existing call sites. The new
copy kernel does not merge those policies. Deferred visibility is scattered
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

Remaining audit work includes full sheet-compaction workspace/destination
propagation, integer-versus-float render metadata separation, generated arena
ABI definitions, resolved batch policy, broader lifetime regions and structural
retry cleanup. Removing the conflict-rank ceiling is a separate behavior change.
