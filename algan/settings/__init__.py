from __future__ import annotations

from algan.settings.abstract_settings import Settings
from algan.settings.computing_settings import ComputingSettings
from algan.settings.path_settings import PathSettings
from algan.settings.raytracing_settings import RayTracingPreset, RayTracingSettings
from algan.settings.root_settings import AlganSettings, SettingsSnapshot
from algan.settings.style_settings import StyleSettings
from algan.settings.video_settings import (
    HD,
    LD,
    MD,
    PREVIEW,
    PRODUCTION,
    SMOKE_TEST,
    THUMBNAIL,
    UHD,
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
    "THUMBNAIL",
    "SMOKE_TEST",
    "PREVIEW",
    "LD",
    "MD",
    "HD",
    "PRODUCTION",
    "UHD",
]
