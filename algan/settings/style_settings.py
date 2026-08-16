"""Default colours and layout style.

``SETTINGS.style`` holds the defaults a Scene picks up when you do not say
otherwise: ``background_color`` (``BLACK``), ``frame`` for the letterbox area
outside the rendered frame, ``text_color`` (``WHITE``), the layout ``buffer``
that ``move_next_to`` and the ``arrange_*`` methods leave between Mobs
(``0.6`` world units), ``fade_out_on_scene_end``, and a ``default_shader`` for
Mobs that set no material of their own.

These are process-wide defaults, and each is overridable closer to the render:
``Scene.set_background_color(...)`` changes one Scene, and
``save_video(background_color=...)`` changes one render.

See :doc:`/advanced_user_tutorials/settings`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from algan.constants.color import BLACK, WHITE, Color
from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings


@dataclass
class StyleSettings(Settings):
    background_color: Color = field(default_factory=lambda: BLACK.clone())
    frame: Color = field(default_factory=lambda: BLACK.clone())
    text_color: Color = field(default_factory=lambda: WHITE.clone())
    buffer: float = 0.6
    fade_out_on_scene_end: bool = False
    default_shader: object | None = None

    def __post_init__(self):
        try:
            buffer = float(self.buffer)
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError("buffer must be a finite number") from exc
        if not math.isfinite(buffer) or buffer < 0:
            raise AlganConfigurationError("buffer must be finite and non-negative")
        if not isinstance(self.fade_out_on_scene_end, bool):
            raise AlganConfigurationError("fade_out_on_scene_end must be a boolean")
        object.__setattr__(self, "buffer", buffer)
