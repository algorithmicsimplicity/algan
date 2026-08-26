"""Runtime compute and memory settings.

``SETTINGS.computing`` holds the knobs that decide how much of the machine a
render may use and how much work is done eagerly -- memory budgets, batch-prep
behaviour, and authoring-time controls.

Device selection is deliberately **not** settable here. The render and animation
devices are read while Torch and Taichi initialize, so
``SETTINGS.computing.set(render_device=...)`` raises with a message pointing at
``ALGAN_RENDER_DEVICE`` rather than silently doing nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from algan.constants.math import GIGABYTES
from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings

_INITIALIZATION_ONLY = {
    "animation_device": "ALGAN_ANIMATION_DEVICE",
    "render_device": "ALGAN_RENDER_DEVICE",
    "render_on_cpu": "ALGAN_RENDER_DEVICE",
}


@dataclass
class ComputingSettings(Settings):
    """Runtime-adjustable memory and authoring controls.

    Device selection is intentionally absent: set ``ALGAN_ANIMATION_DEVICE``
    and ``ALGAN_RENDER_DEVICE`` before importing Algan.

    ``available_memory_override`` pins what
    :func:`~algan.utils.memory_utils.get_num_available_bytes` reports for a
    *measured* device (CUDA, MPS), in bytes; ``None`` (the default) measures
    the device for real. It exists for reproducibility, not for capacity.

    Free device memory is not reproducible -- it shrinks once the Torch and
    Taichi allocators are warm and moves with anything else on the GPU -- and
    the render loop sizes its frame windows from it (the arena, and the merge
    headroom the batch preflight weighs). A different window split carries a
    different set of not-yet-spawned actors and pads the merged arrays to a
    different width, which reorders them and the STBVH; shared-edge depth ties
    then land differently and silhouette pixels move by far more than the
    rounding a pixel-comparison suite budgets for. Pinning the measurement
    makes a render byte-reproducible across processes.

    The value must be affordable on the device: it replaces the measurement
    rather than capping it, so a value larger than the device can supply
    over-commits and falls back to the render loop's out-of-memory retry --
    which re-splits the window and gives up the reproducibility this buys.
    """

    @classmethod
    def _check_keys(cls, kwargs):
        # Devices are chosen while Torch/Taichi initialize, so answer the
        # obvious attempt with the fix rather than "unknown setting".
        for name in kwargs:
            variable = _INITIALIZATION_ONLY.get(name)
            if variable is not None:
                raise AlganConfigurationError(
                    f"{name} is initialization-only; set the {variable} "
                    "environment variable before importing algan"
                )
        super()._check_keys(kwargs)

    animation_memory_fraction: float = 0.15
    rendering_memory_fraction: float = 0.4
    max_animation_batch_size: int = 10000
    max_cpu_memory_used: int = 2 * GIGABYTES
    available_memory_override: int | None = None
    use_torch_scatter: bool = True
    #: Let the batch-prep worker run the render-device projection and scene
    #: merge of batch b+1 while batch b renders, instead of deferring both to
    #: the render thread's arena preflight. The transient-peak predictors must
    #: already be calibrated (the first batch always prepares on the render
    #: thread), overlapped builds skip their peak observations, and the pool
    #: headroom the worker checks against is derated by
    #: ``overlap_pool_headroom_fraction`` to leave room for the concurrent
    #: render. Env override ``ALGAN_PREFETCH_GPU_PREP``. Default off.
    prefetch_gpu_prep: bool = False
    #: Share of :meth:`RenderLoopMixin._gpu_merge_headroom_bytes` an
    #: overlapped (worker-side) projection or merge must fit its predicted
    #: peak in -- the rest of that headroom belongs to the render running
    #: beside it. Only consulted when ``prefetch_gpu_prep`` is active; the
    #: render thread's own preflight keeps the full headroom. Env override
    #: ``ALGAN_OVERLAP_HEADROOM_FRACTION``.
    overlap_pool_headroom_fraction: float = 0.6

    def __post_init__(self):
        for name in ("animation_memory_fraction", "rendering_memory_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise AlganConfigurationError(f"{name} must be in the interval (0, 1]")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.max_animation_batch_size, int)
            or isinstance(self.max_animation_batch_size, bool)
            or self.max_animation_batch_size <= 0
        ):
            raise AlganConfigurationError(
                "max_animation_batch_size must be a positive integer"
            )
        if (
            not isinstance(self.max_cpu_memory_used, int)
            or isinstance(self.max_cpu_memory_used, bool)
            or self.max_cpu_memory_used <= 0
        ):
            raise AlganConfigurationError(
                "max_cpu_memory_used must be a positive integer"
            )
        if self.available_memory_override is not None and (
            not isinstance(self.available_memory_override, int)
            or isinstance(self.available_memory_override, bool)
            or self.available_memory_override <= 0
        ):
            raise AlganConfigurationError(
                "available_memory_override must be a positive integer or None"
            )
        if not isinstance(self.use_torch_scatter, bool):
            raise AlganConfigurationError("use_torch_scatter must be a boolean")
        if not isinstance(self.prefetch_gpu_prep, bool):
            raise AlganConfigurationError("prefetch_gpu_prep must be a boolean")
        fraction = float(self.overlap_pool_headroom_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise AlganConfigurationError(
                "overlap_pool_headroom_fraction must be in the interval (0, 1]"
            )
        object.__setattr__(self, "overlap_pool_headroom_fraction", fraction)
