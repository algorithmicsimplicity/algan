"""Compatibility wrapper for :mod:`algan.settings.style_settings`."""
from algan.settings import SETTINGS
from algan.settings.style_settings import StyleSettings

StyleDefaults = StyleSettings
STYLE_DEFAULTS = SETTINGS.style

__all__ = ["StyleSettings", "StyleDefaults", "STYLE_DEFAULTS"]
