from dataclasses import dataclass
from typing import Tuple

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

    """

    resolution: Tuple[int, int]
    frames_per_second: int = 30
    anti_alias_level: int = 2
    fxaa: bool = False#True
    audio_frames_per_second: int = 44100
    save_image: bool = False


THUMBNAIL = RenderSettings((1280, 720), 1, save_image=True)

SMOKE_TEST = RenderSettings((100, 100), 2, anti_alias_level=1, fxaa=False)

PREVIEW = RenderSettings((704, 396), 10)

LD = RenderSettings((864, 486), 15)

MD = RenderSettings((1280, 720), 30)

HD = RenderSettings((1920, 1080), 30)

PRODUCTION = RenderSettings((2560, 1440), 60)

UHD = RenderSettings((3840, 2160), 60)

QUALITIES: dict[str, dict[str, str | int | None]] = {
    "fourk_quality": UHD,
    "production_quality": PRODUCTION,
    "high_quality": HD,
    "medium_quality": MD,
    "low_quality": LD,
    "example_quality": LD,
}

DEFAULT_QUALITY = "high_quality"
