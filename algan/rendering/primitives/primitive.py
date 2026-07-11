"""Construction/batching base for render primitives.

This package used to be the rasterization pipeline; the renderer is now the
ray tracer under :mod:`algan.rendering.raytracing`, whose primitives subclass
the classes here for their *construction and batching* only (gathering
per-mob tensors into one batched primitive per geometry type). The
rasterization code itself is gone.
"""


class OutOfRenderMemory(Exception):
    """Raised when a frame batch does not fit in the render memory arena;
    the render loop catches it and retries with a halved frame window."""


class RenderPrimitive:
    """Base class for batched render primitives.

    Subclasses are constructed either from per-mob geometry tensors or from a
    ``triangle_collection`` of already-built primitives (the scene batcher
    groups primitives by :meth:`get_batch_identifier` and rebuilds one merged
    primitive per group). The renderer then calls ``project_to_screen`` once
    per batch to shade and pack the geometry, and the memory-accounting
    methods to choose the frame window.
    """

    def get_batch_identifier(self):
        """Key used by the scene batcher: primitives with equal keys are
        merged into one batched primitive of this class."""
        return f"{self.__class__}"

    def get_memory_used_per_timestep(self):
        raise NotImplementedError

    def get_memory_used_for_blending(self, start_ind, end_ind):
        raise NotImplementedError

    def get_memory_used(self, start_ind, end_ind):
        return self.get_memory_used_per_timestep() * (end_ind - start_ind)

    def project_to_screen(self, camera, light_sources):
        raise NotImplementedError
