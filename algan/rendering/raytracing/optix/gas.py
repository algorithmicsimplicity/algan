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
merged scene for the batch: the arena is 40% of the device and builds stay
well inside the rest, and a budget check refuses a batch whose structures
would not (the batch then keeps the software walk). Builds are compacted,
which costs the batch one readback of the compacted sizes.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_uint, c_uint64

import torch

from algan.logging.logger import PERF, get_logger
from algan.rendering.raytracing.optix import _abi as abi
from algan.rendering.raytracing.optix.runtime import CLASS_OPAQUE, CLASS_TRANSLUCENT

logger = get_logger(__name__)

#: Device bytes the uncompacted structures of one batch may take; past it the
#: batch keeps the software walk.
GAS_BUDGET_BYTES = 1 << 30

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


def build_batch_gas(runtime, tri_pos, valid, opaque, casts=None, all_translucent=False):
    """Build one GAS per frame; return a :class:`BatchGas`, or ``None``.

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
    if frames * (out_bytes + _ALIGN) + temp_bytes > GAS_BUDGET_BYTES:
        logger.log(
            PERF,
            f"OptiX: {frames} frames of {num} triangles would need "
            f"{frames * out_bytes / 2**20:.0f} MiB of acceleration structures; "
            "this batch keeps the software traversal",
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
