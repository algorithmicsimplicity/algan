"""Public exception and warning taxonomy for Algan.

The renderer and authoring APIs raise these types when a user-facing
configuration or lifecycle contract is violated.  Keeping the taxonomy small
lets applications catch actionable Algan failures without depending on
implementation-specific exceptions from Torch, Taichi, MoviePy, or FFmpeg.
"""

from __future__ import annotations

import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_stacklevel(default: int = 2) -> int:
    """Frames from the caller out to the first frame outside algan.

    ``warnings.warn(..., stacklevel=N)`` needs a hand-counted N, and the count
    differs between call paths -- ``Scene.save_video`` goes through one more
    frame than ``scene.save_video``. Walking out to the first non-algan frame
    points the warning at the user's own line either way.

    (``warnings.warn``'s ``skip_file_prefixes`` would do this directly, but it
    is Python 3.12+ and Algan supports 3.9.)
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        if not os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
            return level
        frame = frame.f_back
        level += 1
    return default


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


class ModifiedProtectedAttributeError(AlganError, ValueError):
    """Raised when a shader/material is (re)assigned after the mob has spawned."""

    code = "ALGAN_MODIFIED_PROTECTED_ATTRIBUTE"


class TranscriptAudioMismatchError(AlganError, ValueError):
    """Raised when audio duration and transcript mismatch during alignment."""

    code = "ALGAN_TRANSCRIPT_AUDIO_MISMATCH"


class InvalidColorError(AlganError, ValueError):
    """Raised when an invalid color string or value is passed."""

    code = "ALGAN_INVALID_COLOR"


class ContextReuseError(AlganError, RuntimeError):
    """Raised when an animation context object is entered more than once."""

    code = "ALGAN_CONTEXT_REUSE"


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


class NeverSpawnedMobWarning(AlganWarning):
    """Warns that Mobs were authored but never spawned, so they do not appear."""

    code = "ALGAN_NEVER_SPAWNED_MOB"


class DespawnedMobWarning(AlganWarning):
    """Warns that an operation on a despawned Mob cannot bring it back."""

    code = "ALGAN_DESPAWNED_MOB"


__all__ = [
    "AlganError",
    "AlganConfigurationError",
    "UnsupportedFeatureError",
    "HierarchyError",
    "ModifiedProtectedAttributeError",
    "TranscriptAudioMismatchError",
    "InvalidColorError",
    "ContextReuseError",
    "AlganWarning",
    "UnsupportedFeatureWarning",
    "LegacySceneDiscoveryWarning",
    "ApproximationWarning",
    "NeverSpawnedMobWarning",
    "DespawnedMobWarning",
]
