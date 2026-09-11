# Renderer memory ownership: audit implementation

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
into forward arena scratch. The caller's discovery scope must surround the
whole call. Both permutations are included conservatively in the discovery
footprint estimate, even where a native sort falls back to PyTorch. The forward
scope can be overwritten after return without damaging persistent sheet data.

This is **not** a complete arena conversion of compaction: its other tensor
expressions, reductions, and library sorting workspace remain allocator-owned.
A future conversion needs short stage lifetimes rather than accumulating every
old temporary at one bump pointer. The standalone sheet CSR now uses lower
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
so all exits restore both arena ends. The late-built BVH retention floor and
manual sparse-loop lifetime handling remain unchanged. They need a separate
lifetime-region refactor, not removal of the retained floor.

The complete, item-by-item branch checklist and current validation record are
in `reports/renderer_code_audit_status.md` at the repository root.
