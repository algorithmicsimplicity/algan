"""Output format settings and the built-in quality presets.

:class:`VideoSettings` carries resolution, frame rate and supersampling level --
what the encoder is handed, as distinct from what the renderer computes (that is
``SETTINGS.raytracing``).

The presets, in increasing cost:

``SMOKE_TEST``
    32x32 -- fast enough to assert a render ran at all.
``PREVIEW``
    704x396 at 10 fps, no anti-aliasing. The while-you-work setting.
``LD`` / ``MD``
    864x486 at 15 fps; 720p at 30 fps.
``HD``
    1080p at 30 fps.
``PRODUCTION`` / ``UHD``
    1440p and 2160p at 60 fps.
``THUMBNAIL``
    720p, a single frame, heavily anti-aliased -- for
    :meth:`~algan.scene.Scene.save_frame`.

Presets are immutable, so ``HD.set(frames_per_second=60)`` returns a copy and
leaves ``HD`` alone. Pass one to
:meth:`~algan.scene.Scene.save_video` to override quality for a single render.

Two fields carry a short second spelling, because their long names are a
mouthful for something written this often: ``fps``/``FPS`` for
``frames_per_second`` and ``ssaa``/``SSAA`` for ``supersampling``.
They are the same setting -- ``HD.set(fps=60)``
and ``HD.set(frames_per_second=60)`` do the same thing, and reading either
spelling answers with the same value.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings, settings_aliases


def _integer(value):
    """``value`` as an int if it is one -- a NumPy integer too -- but not a bool."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


@settings_aliases(
    fps="frames_per_second",
    FPS="frames_per_second",
    ssaa="supersampling",
    SSAA="supersampling",
)
@dataclass
class VideoSettings(Settings):
    """Video output settings used by :meth:`algan.scene.Scene.save_video`.

    ``frames_per_second`` may also be written ``fps`` or ``FPS``, and
    ``supersampling`` -- how many samples per axis the frame is
    rendered at before being filtered back down to ``resolution`` -- may also be
    written ``ssaa`` or ``SSAA``. The short spellings are accepted for reading,
    for assignment, as constructor keywords and by
    :meth:`~algan.settings.abstract_settings.Settings.set`; the declared names
    are what :meth:`~algan.settings.abstract_settings.Settings.to_dict` and the
    settings snapshot answer with.
    """

    resolution: tuple[int, int]
    frames_per_second: int = 30
    supersampling: int = 2
    fxaa: bool = False
    audio_sample_rate: int = 44100

    def __post_init__(self):
        try:
            width, height = self.resolution
        except Exception as exc:
            raise AlganConfigurationError(
                "resolution must be a pair of positive integers (width, height)"
            ) from exc
        width, height = _integer(width), _integer(height)
        if width is None:
            raise AlganConfigurationError("resolution width must be an integer")
        if height is None:
            raise AlganConfigurationError("resolution height must be an integer")
        if width <= 0 or height <= 0:
            raise AlganConfigurationError("resolution dimensions must be positive")
        object.__setattr__(self, "resolution", (width, height))
        for name in ("frames_per_second", "supersampling", "audio_sample_rate"):
            value = _integer(getattr(self, name))
            if value is None or value <= 0:
                raise AlganConfigurationError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if not isinstance(self.fxaa, bool):
            raise AlganConfigurationError("fxaa must be a boolean")


# Presets are immutable instances of the same class. Their ``set`` method
# returns another preset rather than mutating the shared constant.
def _preset(*args, **kwargs):
    return VideoSettings(*args, **kwargs).as_preset()


THUMBNAIL = _preset((1280, 720), 1, supersampling=4)
SMOKE_TEST = _preset((32, 32), 2, supersampling=1)
PREVIEW = _preset((704, 396), 10, supersampling=1)
LD = _preset((864, 486), 15)
MD = _preset((1280, 720), 30)
HD = _preset((1920, 1080), 30)
PRODUCTION = _preset((2560, 1440), 60)
UHD = _preset((3840, 2160), 60)

# Name -> preset, used by Project's command line to turn --video-settings into
# a preset. Keyed by the name user code writes; the CLI upper-cases what it is
# given. Not part of the public API: user code names the presets directly.
_PRESETS_BY_NAME: dict[str, VideoSettings] = {
    "THUMBNAIL": THUMBNAIL,
    "SMOKE_TEST": SMOKE_TEST,
    "PREVIEW": PREVIEW,
    "LD": LD,
    "MD": MD,
    "HD": HD,
    "PRODUCTION": PRODUCTION,
    "UHD": UHD,
}

# Name -> preset, used by the documentation directive's :quality: option.
# Not part of the public API: user code names the presets directly.
_QUALITIES: dict[str, VideoSettings] = {
    "fourk_quality": UHD,
    "production_quality": PRODUCTION,
    "high_quality": HD,
    "medium_quality": MD,
    "low_quality": LD,
    "example_quality": LD,
}

__all__ = [
    "VideoSettings",
    "THUMBNAIL",
    "SMOKE_TEST",
    "PREVIEW",
    "LD",
    "MD",
    "HD",
    "PRODUCTION",
    "UHD",
]
