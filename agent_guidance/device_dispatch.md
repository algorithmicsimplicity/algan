# Device-count execution and typed arena regions (audit Q1/Q2)

## Enable both

```python
from algan import SETTINGS
SETTINGS.raytracing.experimental.device_dispatch = True
```

Default: `False`. `ALGAN_DEVICE_DISPATCH=1` seeds the same runtime setting before
import. Change it between renders, not while a render is in flight. Turning it
off restores the reference selection/gather/trace path. This implements Q1/Q2
from the audit's Quadrants patch/integration program, not Rank 1/2's raw-fragment
lifetime and general sheet-compaction changes. No performance claim is made.

## Q1: counts, capacity and reusable plans

`device_dispatch.py` defines `DeviceCount` and pointer-free `DispatchPlan`.
Each queue carries capacity-sized buffers and an int32[4] header: published
count, reservations, overflow, error. Counts/offsets stay on the device through
count, hierarchical exclusive scan, fused gather and guarded trace. Converting
a descriptor to a Python int is an error. Scratch is caller-owned arena memory.

The production integration is the deterministic sheet renderer's primary shadow
events. It replaces host `nonzero`, host-sized queue reconstruction and separate
payload gathers. Stable sheet order preserves exact source IDs, frame IDs,
masks (including the sign bit) and reverse sheet-to-event mapping. The reference
arm retains its CUDA source-sort policy, which changes scheduling only.

Normal capacity equals the number of source sheets: every accepted source fits.
Smaller explicit capacities raise `DeviceDispatchOverflow`, never truncate.
The portable trace launches capacity times light count, checked against the
signed-int32 launch limit. It checks status and live count before reading any
payload. Failed batches publish no work and cannot write visibility. One
consolidated status readback after tracing rejects errors before mode-2 shading
can commit the image. Existing continuation-pool overflow/retry is unchanged.

A completion fence covers both native and Torch work before leases are released,
including host exceptions. Unknown completion after a failed fence quarantines
leases instead of allowing memory reuse. This is a correctness boundary, not a
per-stage fence. Submission is capacity-bounded on every supported backend;
there is no claim of hardware indirect dispatch, CUDA Graph capture, or a fully
device-driven continuation loop.

## Q2: typed regions and load metadata

`arena_regions.py` validates storage identity, byte range, dtype, shape, strides,
actual alignment and allocation identity. End offsets are checked, not just
starts. Empty tensors use storage offsets rather than their zero data pointer.
All host metadata arithmetic uses integers; out-of-range layouts fail explicitly.

The complete launch binding set, including ordinary hot arguments, participates
in the alias proof. Read/read aliases are legal. An overlap involving a writer
is rejected even across dtype reinterpretations. Read-only/disjointness facts
belong to exact regions, never the whole mutable arena. Fused gather uses typed
vec3/vec6 payloads while IDs/masks remain integers. Native codegen may scalarize
vectors; no particular generated instruction is promised.

`arena_region_args.py` narrows cold scalar-dtype bindings and rebases exact offset
tables. Immutable layout tables are cached separately from payloads. In the
shadow trace, `ArenaView(..., hoist=True)` materializes offsets, shapes and
strides as scalar locals inside the parallel thread prologue, outside inner
traversal loops. It uses existing compiler argument metadata support, not a new
allocation-wide LLVM invariant or noalias annotation. Whole-arena
`readonly_ndarray_ldg` is not enabled. No new compiler wheel is required.

The seven ordinary hot ray-state arguments in resolve/wavefront kernels are
unchanged. The internal shadow ABI adds a header and template gate; the public
positional calling convention and Metal buffer limit are preserved.

`ManualMemory` starts tracking allocations when the toggle is enabled. Forward
and persistent rewinds invalidate typed handles even after immediate address
reuse. A leased allocation cannot be rewound. Late-bound trace inputs and native
views remain alive through actual completion. An old raw Torch slice without a
typed handle does not record its creation lifetime; bind handles before reuse.

## Scope and correctness checks

CPU, CUDA and adopted MPS/Metal share the queue contract. Selection requires
`taichi_launch_is_local`; other pairings retain the original route. This does
not implement the separate missing AMD/HIP or Vulkan buffer-adoption project.
Hardware validation must be reported only for devices that actually ran tests.

Tests cover zero work, overflow, multilevel scan, reuse, large exact IDs, high
mask bits, optional payloads, invalid flags, failure without visibility writes,
exception fencing, cross-dtype aliases, empty views, nonzero offsets, metadata
mutation, forward/persistent rewind and leases. Shadowed transparent/reflective
smoke renders compare off/on/off and assert the new path actually executed.
