from __future__ import annotations

from dataclasses import dataclass

from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings


@dataclass
class VideoSettings(Settings):
    """Video output settings used by :meth:`algan.scene.Scene.save_video`."""

    resolution: tuple[int, int]
    frames_per_second: int = 30
    anti_alias_level: int = 2
    fxaa: bool = False
    audio_frames_per_second: int = 44100

    def __post_init__(self):
        try:
            width, height = self.resolution
        except Exception as exc:
            raise AlganConfigurationError(
                "resolution must be a pair of positive integers (width, height)"
            ) from exc
        if not isinstance(width, int) or isinstance(width, bool):
            raise AlganConfigurationError("resolution width must be an integer")
        if not isinstance(height, int) or isinstance(height, bool):
            raise AlganConfigurationError("resolution height must be an integer")
        if width <= 0 or height <= 0:
            raise AlganConfigurationError("resolution dimensions must be positive")
        object.__setattr__(self, "resolution", (width, height))
        if (
            not isinstance(self.frames_per_second, int)
            or isinstance(self.frames_per_second, bool)
            or self.frames_per_second <= 0
        ):
            raise AlganConfigurationError(
                "frames_per_second must be a positive integer"
            )
        if (
            not isinstance(self.anti_alias_level, int)
            or isinstance(self.anti_alias_level, bool)
            or self.anti_alias_level <= 0
        ):
            raise AlganConfigurationError("anti_alias_level must be a positive integer")
        if (
            not isinstance(self.audio_frames_per_second, int)
            or isinstance(self.audio_frames_per_second, bool)
            or self.audio_frames_per_second <= 0
        ):
            raise AlganConfigurationError(
                "audio_frames_per_second must be a positive integer"
            )
        if not isinstance(self.fxaa, bool):
            raise AlganConfigurationError("fxaa must be a boolean")


# Presets are immutable instances of the same class. Their ``set`` method
# returns another preset rather than mutating the shared constant.
def _preset(*args, **kwargs):
    return VideoSettings(*args, **kwargs).as_preset()


THUMBNAIL = _preset((1280, 720), 1, anti_alias_level=4)
SMOKE_TEST = _preset((32, 32), 2, anti_alias_level=1)
PREVIEW = _preset((704, 396), 10, anti_alias_level=1)
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
