"""Default colors and layout style.

``SETTINGS.style`` holds the defaults a Scene picks up when you do not say
otherwise: ``background`` (``BLACK``), ``frame`` for the letterbox area
outside the rendered frame, ``text_color`` (``WHITE``), the layout ``buffer``
that ``move_next_to`` and the ``arrange_*`` methods leave between Mobs
(``0.6`` world units), ``fade_out_on_scene_end``, a ``default_material`` for
3-D Mobs that set no material of their own, and ``shape_style_profile``
(``"algan"``), which selects whose per-shape styling defaults the built-in
shapes adopt -- Algan's own, or Manim Community's via
``SETTINGS.style.set(shape_style_profile="manim")``.

These are process-wide defaults, and each is overridable closer to the render:
``Scene.set_background(...)`` changes one Scene, and
``save_video(background=...)`` changes one render.

See :doc:`/advanced_user_tutorials/settings`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from algan.constants.color import BLACK, WHITE, Color
from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings

#: The available shape-style profiles. ``"algan"`` is Algan's own default
#: styling; ``"manim"`` reads each mapped shape's constructor defaults out of
#: the installed manim package (see algan.settings.shape_style_profiles).
SHAPE_STYLE_PROFILES = ("algan", "manim")


@dataclass
class StyleSettings(Settings):
    background: Color = field(default_factory=lambda: BLACK.clone())
    frame: Color = field(default_factory=lambda: BLACK.clone())
    text_color: Color = field(default_factory=lambda: WHITE.clone())
    buffer: float = 0.6
    fade_out_on_scene_end: bool = False
    default_material: object | None = None
    shape_style_profile: str = "algan"

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
        # Duck-typed rather than an isinstance check so this module keeps its
        # no-rendering-at-import-time rule: Material lives under
        # algan.rendering, and settings must stay importable before that.
        if self.default_material is not None and not hasattr(
            self.default_material, "shader"
        ):
            raise AlganConfigurationError(
                "default_material must be a Material instance "
                "(algan.rendering.shaders.materials.Material), which has a "
                ".shader attribute"
            )
        if self.shape_style_profile not in SHAPE_STYLE_PROFILES:
            raise AlganConfigurationError(
                f"shape_style_profile must be one of "
                f"{', '.join(SHAPE_STYLE_PROFILES)}; got "
                f"{self.shape_style_profile!r}"
            )
        if self.shape_style_profile == "manim":
            # Resolving the profile reads the shape defaults out of the
            # installed manim, so enabling pays that import here rather than
            # at the first Mob construction afterwards.
            from algan.settings.shape_style_profiles import (
                _warm_manim_shape_style_cache,
            )

            _warm_manim_shape_style_cache()
