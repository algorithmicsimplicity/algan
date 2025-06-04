from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RenderSettings:
    resolution: Tuple[int, int] # (width, height) in number of pixels
    frames_per_second: int = 30
    anti_alias_level: int = 2
    audio_frames_per_second: int=44100
    save_image: bool = False


EXAMPLE_QUALITY = RenderSettings((854, 480), 30)
PREVIEW = RenderSettings((600, 400), 10)
LD = RenderSettings((854, 480), 15)
SD = RenderSettings((1000, 800), 30)
MD = RenderSettings((1280, 720), 30)
HD = RenderSettings((1920, 1080), 60)
PRODUCTION = RenderSettings((2560, 1440),60)
UHD = RenderSettings((3840, 2160), 60)
THUMBNAIL = RenderSettings((1280, 720), 1, save_image=True)

QUALITIES: dict[str, dict[str, str | int | None]] = {
    "fourk_quality": UHD,
    "production_quality": PRODUCTION,
    "high_quality": HD,
    "medium_quality": MD,
    "low_quality": LD,
    "example_quality": EXAMPLE_QUALITY,
}

DEFAULT_QUALITY = "high_quality"