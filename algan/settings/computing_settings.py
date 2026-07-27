from dataclasses import dataclass
import math

from algan.constants.math import GIGABYTES
from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings


@dataclass
class ComputingSettings(Settings):
    """Runtime-adjustable memory and authoring controls.

    Device selection is intentionally absent: set ``ALGAN_ANIMATION_DEVICE``
    and ``ALGAN_RENDER_DEVICE`` before importing Algan.
    """

    animation_memory_fraction: float = 0.15
    rendering_memory_fraction: float = 0.4
    max_animation_batch_size: int = 10000
    max_cpu_memory_used: int = 2 * GIGABYTES
    use_torch_scatter: bool = True
    allow_save_frame: bool = True

    def __post_init__(self):
        for name in ("animation_memory_fraction", "rendering_memory_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise AlganConfigurationError(f"{name} must be in the interval (0, 1]")
            object.__setattr__(self, name, value)
        if not isinstance(self.max_animation_batch_size, int) or isinstance(
            self.max_animation_batch_size, bool
        ) or self.max_animation_batch_size <= 0:
            raise AlganConfigurationError("max_animation_batch_size must be a positive integer")
        if not isinstance(self.max_cpu_memory_used, int) or isinstance(
            self.max_cpu_memory_used, bool
        ) or self.max_cpu_memory_used <= 0:
            raise AlganConfigurationError("max_cpu_memory_used must be a positive integer")
        if not isinstance(self.use_torch_scatter, bool):
            raise AlganConfigurationError("use_torch_scatter must be a boolean")
        if not isinstance(self.allow_save_frame, bool):
            raise AlganConfigurationError("allow_save_frame must be a boolean")

    # Compatibility names for the old public defaults object.
    @property
    def portion_of_memory_used_for_animating(self):
        return self.animation_memory_fraction

    @portion_of_memory_used_for_animating.setter
    def portion_of_memory_used_for_animating(self, value):
        self.set(animation_memory_fraction=value)

    @property
    def portion_of_memory_used_for_rendering(self):
        return self.rendering_memory_fraction

    @portion_of_memory_used_for_rendering.setter
    def portion_of_memory_used_for_rendering(self, value):
        self.set(rendering_memory_fraction=value)

    @property
    def max_animate_batch_size(self):
        return self.max_animation_batch_size

    @max_animate_batch_size.setter
    def max_animate_batch_size(self, value):
        self.set(max_animation_batch_size=value)
