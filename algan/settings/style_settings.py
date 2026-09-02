"""Default colors and layout style.

``SETTINGS.style`` holds the defaults a Scene picks up when you do not say
otherwise: ``background`` (``BLACK``), ``frame`` for the letterbox area
outside the rendered frame, ``text_color`` (``WHITE``), the layout ``buffer``
that ``move_next_to`` and the ``arrange_*`` methods leave between Mobs
(``0.6`` world units), ``fade_out_on_scene_end``, a ``default_material`` for
3-D Mobs that set no material of their own, ``shape_style_profile``
(``"algan"``), which selects whose per-shape styling defaults the built-in
shapes adopt -- Algan's own, or Manim Community's via
``SETTINGS.style.set(shape_style_profile="manim")`` -- and
``border_placement`` (``"inward"``), which selects whether a filled shape's
stroke is laid inside its outline or straddles it as Manim's does.

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

#: Where a FILLED bezier circuit lays its border relative to the outline.
#: ``"inward"`` (Algan's own) puts the whole stroke inside, so raising
#: ``stroke_width`` eats into the shape and neighbouring glyphs never fuse;
#: ``"centered"`` straddles the outline the way Manim (and SVG's default
#: ``stroke``) does, so half the stroke spills outside and the shape dilates
#: with its stroke width. Unfilled circuits are centred under both -- an open
#: path has no interior to lay a stroke inside of.
BORDER_PLACEMENTS = ("inward", "centered")

#: Manim stroke-width units per Algan unit, used by every conversion in the
#: Manim compatibility layer -- import, export and the shape adapters alike, so
#: a round trip returns the width it started with. Algan's own convention is
#: the round ``2.0``; ``Scene.use_manim_defaults()`` swaps in
#: ``manim_stroke_width_ratio()``, the value that actually draws the same
#: number of pixels Manim draws.
_DEFAULT_MANIM_STROKE_WIDTH_RATIO = 2.0


@dataclass
class StyleSettings(Settings):
    background: Color = field(default_factory=lambda: BLACK.clone())
    frame: Color = field(default_factory=lambda: BLACK.clone())
    text_color: Color = field(default_factory=lambda: WHITE.clone())
    buffer: float = 0.6
    fade_out_on_scene_end: bool = False
    default_material: object | None = None
    shape_style_profile: str = "algan"
    border_placement: str = "inward"
    manim_stroke_width_ratio: float = 2.0

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
        try:
            ratio = float(self.manim_stroke_width_ratio)
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(
                "manim_stroke_width_ratio must be a finite number"
            ) from exc
        if not math.isfinite(ratio) or ratio <= 0:
            raise AlganConfigurationError(
                "manim_stroke_width_ratio must be finite and positive"
            )
        object.__setattr__(self, "manim_stroke_width_ratio", ratio)
        if self.border_placement not in BORDER_PLACEMENTS:
            raise AlganConfigurationError(
                f"border_placement must be one of "
                f"{', '.join(BORDER_PLACEMENTS)}; got "
                f"{self.border_placement!r}"
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
