from dataclasses import dataclass, field
import math

from algan.constants.color import BLACK, WHITE, Color
from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings

__all__ = ["StyleDefaults", "STYLE_DEFAULTS"]


@dataclass
class StyleDefaults(Settings):
    background_color: Color = field(default_factory=lambda: BLACK.clone())
    frame: Color = field(default_factory=lambda: BLACK.clone())
    text_color: Color = field(default_factory=lambda: WHITE.clone())
    buffer: float = 0.6
    fade_out_on_scene_end: bool = False

    def __post_init__(self):
        try:
            buffer = float(self.buffer)
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError("buffer must be a finite number") from exc
        if not math.isfinite(buffer) or buffer < 0:
            raise AlganConfigurationError("buffer must be finite and non-negative")
        if not isinstance(self.fade_out_on_scene_end, bool):
            raise AlganConfigurationError(
                "fade_out_on_scene_end must be a boolean"
            )
        object.__setattr__(self, "buffer", buffer)


STYLE_DEFAULTS = StyleDefaults()
