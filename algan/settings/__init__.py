"""The process-global settings root, :data:`SETTINGS`.

Everything configurable at runtime hangs off one object, grouped into sections
by what it affects: ``SETTINGS.video`` (resolution, frame rate, anti-aliasing),
``SETTINGS.style`` (default colours, layout buffer, scene-end fade),
``SETTINGS.paths`` (output and cache locations), ``SETTINGS.computing`` (memory
budgets and authoring controls) and ``SETTINGS.raytracing`` (what the renderer
produces).

Sections have **stable identity**: mutate them in place with
``SETTINGS.video.set(HD)``, never ``SETTINGS.video = HD``, so that code holding a
reference to a section keeps seeing live values.

Device selection is not here. ``ALGAN_RENDER_DEVICE`` and
``ALGAN_ANIMATION_DEVICE`` are read while Torch and Taichi initialize, so they
must be set in the environment before ``import algan`` and have no runtime object
to assign to.

See :doc:`/advanced_user_tutorials/settings`.
"""

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
