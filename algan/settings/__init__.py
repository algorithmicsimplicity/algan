from algan.settings.abstract_settings import Settings
from algan.settings.computing_settings import ComputingSettings
from algan.settings.path_settings import PathSettings
from algan.settings.raytracing_settings import RayTracingPreset, RayTracingSettings
from algan.settings.root_settings import AlganSettings, SettingsSnapshot
from algan.settings.style_settings import StyleSettings
from algan.settings.video_settings import (
    DEFAULT_QUALITY,
    HD,
    LD,
    MD,
    PREVIEW,
    PRODUCTION,
    QUALITIES,
    SMOKE_TEST,
    THUMBNAIL,
    UHD,
    RenderSettings,
    VideoSettings,
)

SETTINGS = AlganSettings()

__all__ = [
    "SETTINGS",
    "Settings",
    "AlganSettings",
    "SettingsSnapshot",
    "ComputingSettings",
    "PathSettings",
    "StyleSettings",
    "RayTracingSettings",
    "RayTracingPreset",
    "VideoSettings",
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
