"""Public exception and warning taxonomy for Algan.

The renderer and authoring APIs raise these types when a user-facing
configuration or lifecycle contract is violated.  Keeping the taxonomy small
lets applications catch actionable Algan failures without depending on
implementation-specific exceptions from Torch, Taichi, MoviePy, or FFmpeg.
"""


class AlganError(Exception):
    """Base class for user-facing Algan exceptions."""

    code = "ALGAN_ERROR"


class AlganConfigurationError(AlganError, ValueError):
    """Raised when a supplied setting or render configuration is invalid."""

    code = "ALGAN_CONFIGURATION_ERROR"


class UnsupportedFeatureError(AlganConfigurationError):
    """Raised when the selected renderer cannot honor requested features."""

    code = "ALGAN_UNSUPPORTED_FEATURE"


class HierarchyError(AlganError, ValueError):
    """Raised when a Mob hierarchy mutation would create an invalid graph."""

    code = "ALGAN_INVALID_HIERARCHY"


class AlganWarning(UserWarning):
    """Base class for user-facing Algan warnings."""

    code = "ALGAN_WARNING"


class UnsupportedFeatureWarning(AlganWarning):
    """Warns that a renderer cannot honor one or more requested features."""

    code = "ALGAN_UNSUPPORTED_FEATURE"


class LegacySceneDiscoveryWarning(AlganWarning):
    """Warns that render_all_funcs fell back to implicit function scanning."""

    code = "ALGAN_LEGACY_SCENE_DISCOVERY"


class ApproximationWarning(AlganWarning):
    """Warns that an API uses an explicitly documented approximation."""

    code = "ALGAN_APPROXIMATION"


__all__ = [
    "AlganError",
    "AlganConfigurationError",
    "UnsupportedFeatureError",
    "HierarchyError",
    "AlganWarning",
    "UnsupportedFeatureWarning",
    "LegacySceneDiscoveryWarning",
    "ApproximationWarning",
]
