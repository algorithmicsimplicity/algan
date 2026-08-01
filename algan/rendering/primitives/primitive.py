"""Construction/batching base for render primitives.

This package used to be the rasterization pipeline; the renderer is now the
ray tracer under :mod:`algan.rendering.raytracing`, whose primitives subclass
the classes here for their *construction and batching* only (gathering
per-mob tensors into one batched primitive per geometry type). The
rasterization code itself is gone.
"""
from __future__ import annotations

import copy

import torch


def _slice_frame_value(value, start, end, total_frames):
    """Slice a source-primitive value whose leading axis is batch time.

    Static values conventionally carry a singleton leading axis and are
    shared.  Containers are preserved because shader parameters are stored as
    a list of tensors.  The returned tensors are views: arena prefix selection
    only uses this path when projection uploads them to a different device, so
    projection cannot mutate the pristine fetched batch.
    """
    if torch.is_tensor(value):
        if value.ndim > 0 and int(value.shape[0]) == int(total_frames):
            return value[int(start):int(end)]
        return value
    if isinstance(value, list):
        return [_slice_frame_value(v, start, end, total_frames) for v in value]
    if isinstance(value, tuple):
        return tuple(
            _slice_frame_value(v, start, end, total_frames) for v in value
        )
    return value


class OutOfRenderMemory(Exception):
    """Raised when a frame batch does not fit in the render memory arena;
    the render loop catches it and retries with a halved frame window.
    """


class RenderPrimitive:
    """Base class for batched render primitives.

    Subclasses are constructed either from per-mob geometry tensors or from a
    ``triangle_collection`` of already-built primitives (the scene batcher
    groups primitives by :meth:`get_batch_identifier` and rebuilds one merged
    primitive per group). The renderer then calls ``project_to_screen`` once
    per batch to shade and pack the geometry, and the memory-accounting
    methods to choose the frame window.
    """

    # Subclasses list source tensors whose leading axis is animation time.
    # This supports render-arena prefix selection without rematerializing the
    # timeline or rebuilding source primitives for every binary-search probe.
    frame_dependent_source_attrs = ()

    def slice_time_window(self, start, end, total_frames):
        """Return a shallow primitive copy restricted to a frame window.

        Only source-geometry attributes declared by the subclass are sliced.
        Topology, shader objects, texture metadata and scene/memory handles are
        shared.  Packed ``_rt_*`` state is deliberately cleared: each candidate
        is projected and merged independently during arena preflight.
        """
        start = int(start)
        end = int(end)
        total_frames = int(total_frames)
        if not (0 <= start < end <= total_frames):
            raise ValueError(
                f"invalid primitive frame window {start}:{end} for "
                f"{total_frames} frames"
            )

        result = copy.copy(self)
        for name in self.frame_dependent_source_attrs:
            if hasattr(self, name):
                setattr(
                    result, name,
                    _slice_frame_value(
                        getattr(self, name), start, end, total_frames
                    ),
                )
        for name in tuple(vars(result)):
            if name.startswith("_rt_"):
                delattr(result, name)
        return result

    def get_batch_identifier(self):
        """Key used by the scene batcher: primitives with equal keys are
        merged into one batched primitive of this class.
        """
        return f"{self.__class__}"

    def project_to_screen(self, camera, light_sources):
        raise NotImplementedError
