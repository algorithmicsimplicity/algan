"""Triangle acceleration structures for one render batch.

One OptiX GAS per batch frame, built from the merged scene's own buffers:

* vertices are ``tri_pos[f % Tp]``, read in place (three ``float3`` per row);
* the index buffer turns every primitive the frame cannot see -- outside its
  lifespan, faded out (``tri_frame_valid``), or, for shadow rays, declared
  ``casts_shadows = False`` -- into a degenerate triangle, which no ray hits,
  so the OptiX primitive index *is* Algan's primitive index;
* a per-primitive SBT index puts each one in a class: opaque at that frame
  (``tri_frame_opaque``: any-hit disabled, the hit is final in hardware) or
  translucent (an any-hit program records it).

Frames share a GAS when nothing that decides it changes over the batch (every
source table deduplicated to one frame); otherwise there is one per frame.

The GASes are ordinary PyTorch allocations, not arena views, kept on the
merged scene for the batch. One budget covers all of a batch's structures
(the shadow GASes and, where non-casters make it separate, the gather's): at
most ``GAS_BUDGET_BYTES``, and at most ``FREE_SHARE`` of the device memory
free outside PyTorch's live tensors when a build starts, counting each
build's peak -- every uncompacted structure, the shared scratch and the
compacted copies made before the uncompacted ones go. A batch past it keeps
the software walk. The render loop's arena preflight asks the same question
of a frame window before it renders one (:func:`preflight_bytes`), so a window
whose structures would not fit is shrunk rather than rendered in software.
Builds are compacted, which costs the batch one readback of the compacted
sizes.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_uint, c_uint64

import torch

from algan.logging.logger import PERF, get_logger
from algan.rendering.raytracing.optix import _abi as abi
from algan.rendering.raytracing.optix.runtime import CLASS_OPAQUE, CLASS_TRANSLUCENT

logger = get_logger(__name__)

#: Device bytes all of one batch's structures may take at their builds' peak.
GAS_BUDGET_BYTES = 1 << 30

#: Share of the device memory free outside PyTorch's live tensors (driver free
#: plus PyTorch's cached, unallocated blocks) a batch's structures may take;
#: the rest stays for the next batch's merge and everything else outside the
#: arena.
FREE_SHARE = 0.5

#: Build peak per triangle and GAS frame, and per triangle of the scratch one
#: build reuses, for estimates made before any runtime exists (the arena
#: preflight). Measured with OptiX 9.1 on an RTX 5080: the uncompacted
#: structure takes 53.6 bytes a triangle, its compacted copy ~22, the scratch
#: 44.6; the uncompacted size bounds the copy here.
PEAK_BYTES_PER_TRIANGLE_FRAME = 112
SCRATCH_BYTES_PER_TRIANGLE = 48
_FRAME_OVERHEAD_BYTES = 16 * 1024

_ALIGN = abi.OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT


class BatchGas:
    """The GASes of one batch and one ray kind, and the handle table rays index."""

    def __init__(
        self, handles, buffers, frames, triangles, build_ms=None, all_translucent=False
    ):
        #: ``int64`` device tensor of OptiX traversable handles, one per frame
        #: (or one shared by every frame): rays use ``handles[f % len]``.
        self.handles = handles
        self.buffers = buffers
        self.frames = frames
        self.triangles = triangles
        self.build_ms = build_ms
        #: Every primitive in the translucent class (shadow GASes under a
        #: ``min_alpha`` or surface cap the opaque shortcut is not exact for).
        self.all_translucent = all_translucent

    @property
    def num_handles(self):
        return int(self.handles.numel())

    def nbytes(self):
        return sum(int(b.numel()) for b in self.buffers)


def _aligned(buffer, nbytes):
    """A pointer into ``buffer`` aligned for an acceleration structure."""
    pointer = buffer.data_ptr()
    pad = (-pointer) % _ALIGN
    assert pad + nbytes <= buffer.numel()
    return pointer + pad


def _alloc(nbytes, device):
    return torch.empty(max(1, nbytes) + _ALIGN, dtype=torch.uint8, device=device)


def _frames_of(*tables):
    """The batch's GAS frame count: 1 when every table is one frame."""
    counts = {int(t.shape[0]) for t in tables}
    frames = max(counts)
    if counts - {1, frames}:
        return None
    return frames


def device_free_bytes(device):
    """Device memory free outside PyTorch's live tensors.

    What the driver has free plus PyTorch's cached blocks no tensor holds.
    """
    device = torch.device(device)
    free, _total = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return int(free) + max(0, int(cached))


def batch_budget(device, held_bytes=0):
    """Bytes a batch's next build may take at its peak."""
    return max(
        0,
        min(
            GAS_BUDGET_BYTES - int(held_bytes),
            int(FREE_SHARE * device_free_bytes(device)),
        ),
    )


def preflight_bytes(merged, num_triangles):
    """Estimated build peak of a batch's shadow GASes, for the arena preflight.

    Frame count from the merged tables' frame axes (one GAS when every table
    is one frame), bytes from the measured per-triangle rates above; ``0``
    when the batch has nothing to build.
    """
    num = int(num_triangles)
    tables = [
        merged.get(key)
        for key in ("tri_pos", "tri_frame_valid", "tri_frame_opaque", "tri_frame_casts")
    ]
    tables = [t for t in tables if t is not None]
    if num <= 0 or not tables:
        return 0
    frames = _frames_of(*tables) or max(int(t.shape[0]) for t in tables)
    return (
        frames * (num * PEAK_BYTES_PER_TRIANGLE_FRAME + _FRAME_OVERHEAD_BYTES)
        + num * SCRATCH_BYTES_PER_TRIANGLE
    )


def build_batch_gas(
    runtime, tri_pos, valid, opaque, casts=None, all_translucent=False, held_bytes=0
):
    """Build one GAS per frame; return a :class:`BatchGas`, or ``None``.

    ``held_bytes`` is what this batch's structures already hold (the shadow
    GASes, when the gather builds its own): the budget is the batch's.

    ``None`` -- the batch keeps the software walk -- also when the build
    fails: out of device memory (the structures live outside the arena, so the
    render's own out-of-memory retry, which shrinks the frame window, would not
    help) or refused by the driver. Logged once.
    """
    global _BUILD_FAILURE_LOGGED
    try:
        return _build_batch_gas(
            runtime,
            tri_pos,
            valid,
            opaque,
            casts=casts,
            all_translucent=all_translucent,
            held_bytes=held_bytes,
        )
    except (abi.OptixError, torch.OutOfMemoryError) as exc:
        if not _BUILD_FAILURE_LOGGED:
            _BUILD_FAILURE_LOGGED = True
            logger.warning(
                "OptiX: building this batch's acceleration structures failed, so "
                "it renders with the software traversal: %s",
                exc,
            )
        return None


_BUILD_FAILURE_LOGGED = False


def _build_batch_gas(
    runtime, tri_pos, valid, opaque, casts=None, all_translucent=False, held_bytes=0
):
    """:func:`build_batch_gas` without the fallback.

    ``tri_pos`` is ``[Tp, N, 9]`` float32; ``valid`` / ``opaque`` / ``casts``
    are ``[T?, N]`` bool tables (one frame or the batch's). ``casts`` given
    removes non-casters (shadow rays); ``all_translucent`` puts every
    primitive in the translucent class (an any-hit sees every hit).
    """
    device = tri_pos.device
    num = int(tri_pos.shape[1])
    if num <= 0 or num > runtime.max_primitives > 0:
        return None
    tables = [tri_pos, valid, opaque] + ([casts] if casts is not None else [])
    frames = _frames_of(*tables)
    if frames is None:
        return None
    if tri_pos.dtype != torch.float32 or not tri_pos.is_contiguous():
        tri_pos = tri_pos.float().contiguous()

    flags = (c_uint * 2)()
    flags[CLASS_OPAQUE] = (
        abi.OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
        | abi.OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
    )
    flags[CLASS_TRANSLUCENT] = abi.OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
    options = abi.OptixAccelBuildOptions(
        buildFlags=abi.OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        | abi.OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        operation=abi.OPTIX_BUILD_OPERATION_BUILD,
    )
    vertex_slot = (c_uint64 * 1)()

    def build_input(vertices_ptr, index_ptr, sbt_ptr):
        vertex_slot[0] = vertices_ptr
        tri = abi.OptixBuildInputTriangleArray(
            vertexBuffers=ctypes.cast(vertex_slot, POINTER(c_uint64)),
            numVertices=3 * num,
            vertexFormat=abi.OPTIX_VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes=12,
            indexBuffer=index_ptr,
            numIndexTriplets=num,
            indexFormat=abi.OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            indexStrideInBytes=12,
            preTransform=0,
            flags=ctypes.cast(flags, POINTER(c_uint)),
            numSbtRecords=2,
            sbtIndexOffsetBuffer=sbt_ptr,
            sbtIndexOffsetSizeInBytes=1,
            sbtIndexOffsetStrideInBytes=1,
        )
        built = abi.OptixBuildInput(type=abi.OPTIX_BUILD_INPUT_TYPE_TRIANGLES)
        built.triangleArray = tri
        return built

    # Sizes depend on counts and formats only; any aligned pointers will do.
    anchor = tri_pos.data_ptr()
    sizes = runtime.accel_sizes(build_input(anchor, anchor, anchor), options)
    out_bytes = int(sizes.outputSizeInBytes)
    temp_bytes = int(sizes.tempSizeInBytes)
    # The peak: every uncompacted structure, the shared scratch, and the
    # compacted copies (no larger than the uncompacted ones) made before the
    # uncompacted ones are released.
    peak = frames * 2 * (out_bytes + _ALIGN) + temp_bytes + _ALIGN
    budget = batch_budget(device, held_bytes)
    if peak > budget:
        logger.log(
            PERF,
            f"OptiX: {frames} frames of {num} triangles would need "
            f"{peak / 2**20:.0f} MiB at their build's peak, past the batch's "
            f"{budget / 2**20:.0f} MiB; this batch keeps the software traversal",
        )
        return None

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    base = torch.arange(3 * num, dtype=torch.int32, device=device).view(num, 3)
    degenerate = base[:, :1].expand(num, 3)
    temp = _alloc(temp_bytes, device)
    temp_ptr = _aligned(temp, temp_bytes)
    compacted_sizes = torch.zeros(frames, dtype=torch.int64, device=device)
    uncompacted = []
    handles = []
    for f in range(frames):
        hidden = ~valid[f % valid.shape[0]]
        if casts is not None:
            hidden = hidden | ~casts[f % casts.shape[0]]
        index = torch.where(hidden.unsqueeze(-1), degenerate, base).contiguous()
        if all_translucent:
            classes = torch.full(
                (num,), CLASS_TRANSLUCENT, dtype=torch.uint8, device=device
            )
        else:
            classes = torch.where(
                opaque[f % opaque.shape[0]],
                torch.tensor(CLASS_OPAQUE, dtype=torch.uint8, device=device),
                torch.tensor(CLASS_TRANSLUCENT, dtype=torch.uint8, device=device),
            ).contiguous()
        out = _alloc(out_bytes, device)
        emit = abi.OptixAccelEmitDesc(
            result=compacted_sizes.data_ptr() + 8 * f,
            type=abi.OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
        )
        handle = runtime.accel_build(
            build_input(
                tri_pos[f % tri_pos.shape[0]].data_ptr(),
                index.data_ptr(),
                classes.data_ptr(),
            ),
            options,
            temp_ptr,
            temp_bytes,
            _aligned(out, out_bytes),
            out_bytes,
            emit,
        )
        # The index and class tables are read by the build, in stream order;
        # the caching allocator hands their blocks out again only to later
        # work on the same stream, so dropping them here is safe.
        uncompacted.append(out)
        handles.append(handle)
    # One readback for the batch: the compacted sizes.
    compact = compacted_sizes.tolist()
    buffers = []
    final = []
    for f in range(frames):
        size = int(compact[f])
        if 0 < size < out_bytes:
            out = _alloc(size, device)
            final.append(runtime.accel_compact(handles[f], _aligned(out, size), size))
            buffers.append(out)
        else:
            final.append(handles[f])
            buffers.append(uncompacted[f])
    stop.record()
    del temp, uncompacted
    table = (
        torch.tensor(
            [h - (1 << 64) if h >= 1 << 63 else h for h in final],
            dtype=torch.int64,
        )
        .pin_memory()
        .to(device, non_blocking=True)
    )
    return BatchGas(table, buffers, frames, num, (start, stop), all_translucent)
