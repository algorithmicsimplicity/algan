from dataclasses import dataclass
from typing import Tuple

from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import Settings


@dataclass
class RenderSettings(Settings):
    """Contains all the settings for a rendering operation, as used in :func:`~.render_to_file` .

    Parameters
    ----------
    resolution
        Pair of (width, height), giving the number of pixels in the video frame.
    frames_per_second
        Frames per second in the video.
    anti_alias_level
        To perform anti-aliasing video is rendered at a resolution equal to
        the resolution times `anti_alias_level`, then average pooled back down
        to the original image. This results in smoother edge transitions,
        at a cost of anti_alias_level^2 factor increase in computation.

    Examples
    --------
    Render with custom settings, (1000, 1000) resolution at 100 frames_per_second,
    and anti alias level 1 (no anti alias).

    .. code-block:: python

        render_to_file(render_settings=RenderSettings((1000, 1000), 100, 1))
        render_to_file(render_settings=HD.set_frames_per_second(60))

    """

    resolution: Tuple[int, int]
    frames_per_second: int = 30
    anti_alias_level: int = 1
    fxaa: bool = False
    audio_frames_per_second: int = 44100
    save_image: bool = False

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
        # Store a canonical immutable pair even when callers supplied a list.
        object.__setattr__(self, "resolution", (width, height))
        if self.frames_per_second <= 0:
            raise AlganConfigurationError("frames_per_second must be positive")
        if not isinstance(self.anti_alias_level, int) or isinstance(
            self.anti_alias_level, bool
        ):
            raise AlganConfigurationError("anti_alias_level must be an integer")
        if self.anti_alias_level <= 0:
            raise AlganConfigurationError("anti_alias_level must be positive")
        if not isinstance(self.audio_frames_per_second, int) or isinstance(
            self.audio_frames_per_second, bool
        ):
            raise AlganConfigurationError(
                "audio_frames_per_second must be an integer"
            )
        if self.audio_frames_per_second <= 0:
            raise AlganConfigurationError(
                "audio_frames_per_second must be positive"
            )
        if not isinstance(self.fxaa, bool):
            raise AlganConfigurationError("fxaa must be a boolean")
        if not isinstance(self.save_image, bool):
            raise AlganConfigurationError("save_image must be a boolean")


THUMBNAIL = RenderSettings((1280, 720), 1, anti_alias_level=4, save_image=True)
SMOKE_TEST = RenderSettings((32, 32), 2, anti_alias_level=1)
PREVIEW = RenderSettings((704, 396), 10, anti_alias_level=1)
LD = RenderSettings((864, 486), 15)
MD = RenderSettings((1280, 720), 30)
HD = RenderSettings((1920, 1080), 30)
PRODUCTION = RenderSettings((2560, 1440), 60)
UHD = RenderSettings((3840, 2160), 60)

QUALITIES: dict[str, RenderSettings] = {
    "fourk_quality": UHD,
    "production_quality": PRODUCTION,
    "high_quality": HD,
    "medium_quality": MD,
    "low_quality": LD,
    "example_quality": LD,
}

DEFAULT_QUALITY = "high_quality"

__all__ = [
    "RenderSettings",
    "THUMBNAIL",
    "SMOKE_TEST",
    "PREVIEW",
    "LD",
    "MD",
    "HD",
    "PRODUCTION",
    "UHD",
    "QUALITIES",
    "DEFAULT_QUALITY",
]
