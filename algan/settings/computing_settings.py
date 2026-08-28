"""Runtime compute and memory settings.

``SETTINGS.computing`` holds the knobs that decide how much of the machine a
render may use and how much work is done eagerly -- memory budgets, batch-prep
behaviour, the render device, and authoring-time controls.

The **animation** device is still initialization-only: it is where every Mob's
authoring state is allocated, from the first ``Square()`` onward, so by the time
a script could ask for it the tensors already exist.
``SETTINGS.computing.set(animation_device=...)`` raises with a message pointing
at ``ALGAN_ANIMATION_DEVICE`` rather than silently doing nothing.

The **render** device is settable, because nothing that outlives a render is
allocated on it: the arena is built per job, every cross-render geometry cache
is keyed by device, and Taichi's arch is re-selected at render start (see
``taichi_runtime.ensure_taichi_for_render``). The one exception is a wide
attribute -- a texture, which materializes its frame window on the render device
-- and :func:`~algan.animation_timeline.timeline.wide_attribute_device_pin`
refuses a change once one exists rather than letting the two disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from algan.constants.math import GIGABYTES
from algan.errors import AlganConfigurationError
from algan.settings._startup import _DEFAULT_RENDER_DEVICE, coerce_device
from algan.settings.abstract_settings import Settings

_INITIALIZATION_ONLY = {
    "animation_device": "ALGAN_ANIMATION_DEVICE",
}

#: Names that used to exist here, and the field that replaced them.
_RENAMED = {
    "render_on_cpu": "render_device",
}


def _check_render_device_change_allowed(current, requested):
    """Raise unless the render device can still be changed.

    Two things stand in the way. The first is a render **in progress**: the
    batch-prep worker launches kernels on its own thread, so a change mid-job
    could have it re-initialize Taichi -- discarding every compiled kernel --
    while the render thread is inside one.

    The second outlives a render: a **wide attribute**.
    ``AttributeTimeline`` decides at construction whether an attribute is wide
    enough to materialize its frame window on the render device, and a texture
    is, so a ``Surface`` created before the change holds a decision made for the
    old device. Nothing downstream re-asks. Rather than migrate buffers whose
    whole purpose is to be large, refuse: the device is a property of the
    process and belongs at the top of the script, before any Mob exists.
    """
    from algan.animation_timeline.timeline import wide_attribute_device_pin
    from algan.rendering.taichi_runtime import render_is_active

    if render_is_active():
        raise AlganConfigurationError(
            f"render_device cannot change from {current} to {requested} while a "
            "render is in progress: the batch-prep worker is launching kernels "
            "on the arch this would replace. Set it before save_video() or "
            "save_frame(), not from inside an updater or a post-process."
        )
    pin = wide_attribute_device_pin()
    if pin is None:
        return
    raise AlganConfigurationError(
        f"render_device cannot change from {current} to {requested}: a wide "
        f"attribute (a texture) already materializes on {pin}, and its buffers "
        "are placed when the Mob is created. Set the render device before "
        "creating any textured Mob -- at the top of the script, or with "
        "ALGAN_RENDER_DEVICE -- or start a fresh Scene with "
        "SceneManager.reset()."
    )


@dataclass
class ComputingSettings(Settings):
    """Runtime-adjustable memory, device and authoring controls.

    ``render_device`` is settable here; the animation device is not -- set
    ``ALGAN_ANIMATION_DEVICE`` before importing Algan.

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
        # The animation device is chosen while Torch initializes, so answer the
        # obvious attempt with the fix rather than "unknown setting".
        for name in kwargs:
            variable = _INITIALIZATION_ONLY.get(name)
            if variable is not None:
                raise AlganConfigurationError(
                    f"{name} is initialization-only; set the {variable} "
                    "environment variable before importing algan"
                )
            replacement = _RENAMED.get(name)
            if replacement is not None:
                raise AlganConfigurationError(
                    f"There is no {name} setting; set {replacement} instead, "
                    f"e.g. SETTINGS.computing.set({replacement}='cpu')"
                )
        super()._check_keys(kwargs)

    def __setattr__(self, name, value):
        # Direct assignment reaches ``object.__setattr__`` for every other
        # field, which is tolerable for a number and not for this: an
        # unvalidated device, or one changed after a texture pinned it, renders
        # on the wrong hardware or crashes deep inside a batch. Route it
        # through ``set`` so assignment and ``set(render_device=...)`` are the
        # same operation. ``set`` writes with ``object.__setattr__``, so this
        # cannot recurse, and the ``hasattr`` guard lets construction through.
        if (
            name == "render_device"
            and not getattr(self, "_is_preset", False)
            and hasattr(self, "render_device")
        ):
            self.set(render_device=value)
            return
        super().__setattr__(name, value)

    def set(self, source=None, **kwargs):
        """Apply settings, refusing a render-device change that is too late.

        The check runs *before* the base class writes anything: a rejected
        change must leave the section exactly as it was, not half-applied.
        """
        if "render_device" in kwargs:
            requested = kwargs["render_device"]
        elif source is not None:
            requested = getattr(source, "render_device", self.render_device)
        else:
            requested = self.render_device
        requested = coerce_device(requested, "render_device")
        if not self.is_preset and requested != self.render_device:
            _check_render_device_change_allowed(self.render_device, requested)
        return super().set(source, **kwargs)

    #: Where render primitives are built and the ray tracer runs. Accepts a
    #: ``torch.device``, a device string, or ``'auto'`` (what
    #: ``ALGAN_RENDER_DEVICE`` defaults to), and is normalized to a
    #: ``torch.device``. Read it through
    #: :func:`algan.settings._startup.render_device`, never by binding it at
    #: import. Changing it re-selects Taichi's arch at the next render, which
    #: costs one kernel-preparation pass (the compiled kernels of the old arch
    #: are discarded), so switch once at the top of a script rather than
    #: between renders.
    render_device: torch.device = field(default_factory=lambda: _DEFAULT_RENDER_DEVICE)
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
        object.__setattr__(
            self, "render_device", coerce_device(self.render_device, "render_device")
        )
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
