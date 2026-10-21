"""Public exception and warning taxonomy for Algan.

The renderer and authoring APIs raise these types when a user-facing
configuration or lifecycle contract is violated.  Keeping the taxonomy small
lets applications catch actionable Algan failures without depending on
implementation-specific exceptions from Torch, Taichi, MoviePy, or FFmpeg.
"""

from __future__ import annotations

import os
import sys
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _in_algan(filename: str) -> bool:
    """Whether code from ``filename`` is part of this package.

    Inside its directory, that is, not merely named like it: a script in an
    ``algan_scenes`` folder beside an editable checkout is the user's.
    """
    return os.path.abspath(filename).startswith(_PACKAGE_DIR + os.sep)


def _user_stacklevel(default: int = 2) -> int:
    """Frames from the caller out to the first frame outside algan.

    ``warnings.warn(..., stacklevel=N)`` needs a hand-counted N, and the count
    differs between call paths -- ``Scene.save_video`` goes through one more
    frame than ``scene.save_video``. Walking out to the first non-algan frame
    points the warning at the user's own line either way.

    (``warnings.warn``'s ``skip_file_prefixes`` would do this directly, but it
    is Python 3.12+ and Algan supports 3.10.)
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        if not _in_algan(frame.f_code.co_filename):
            return level
        frame = frame.f_back
        level += 1
    return default


def _user_location():
    """The first frame outside algan, as ``(filename, lineno, module_globals)``.

    For a warning that can only be decided after the user's line has returned
    -- once an enclosing animation context has exited and its timestamps are
    final -- but should still point at that line. Pass it to :func:`_warn_at`.
    """
    frame = sys._getframe(1)
    while frame is not None and _in_algan(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None:
        return None
    return frame.f_code.co_filename, frame.f_lineno, frame.f_globals


def _warn_at(location, message, category):
    """``warnings.warn`` attributed to a :func:`_user_location`.

    Uses the module's own warning registry, exactly as ``warnings.warn`` with
    a ``stacklevel`` pointing at that frame would, so the usual filters and
    the once-per-location default apply unchanged.
    """
    if location is None:
        warnings.warn(message, category, stacklevel=2)
        return
    filename, lineno, module_globals = location
    warnings.warn_explicit(
        message,
        category,
        filename,
        lineno,
        module=module_globals.get("__name__"),
        registry=module_globals.setdefault("__warningregistry__", {}),
        module_globals=module_globals,
    )


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


class AudioTranscriptMismatchError(AlganError, ValueError):
    """Raised when audio runtime and transcript mismatch during alignment."""

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


class NeverVisibleMobWarning(AlganWarning):
    """Warns that a Mob is despawned before its animated spawn could show it.

    Typically ``spawn()`` and ``despawn()`` written side by side in one
    :class:`~algan.animation_timeline.animation_contexts.Sync`: both start
    when the block does, so the exit fades the Mob out while the entrance
    fades it in, and it is never drawn.
    """

    code = "ALGAN_NEVER_VISIBLE_MOB"


class AnimatedBeforeSpawnWarning(AlganWarning):
    """Warns that an animated function took effect at once on a Mob not yet
    spawned, and the Mob was then spawned inside the same animation block.

    Nothing is recorded for a Mob with nothing on screen: the call is applied
    immediately, so the Mob appears already in its final state. Spawn it before
    the call -- or call it after the block -- to see the change animate.
    """

    code = "ALGAN_ANIMATED_BEFORE_SPAWN"


class HierarchyChangedDuringUpdaterWarning(AlganWarning):
    """Warns that a hierarchy change reaches back over a live updater's frames.

    A recorded animation resolves which Mobs it covers once, when it is
    recorded. An updater does not: it is re-run for every frame it covers, and
    re-resolves its subtree against the hierarchy as it stands when the frames
    are rendered. So attaching or detaching a Mob inside a subtree a live
    updater drives changes the frames that updater already covered, including
    frames before the line that made the change.
    """

    code = "ALGAN_HIERARCHY_CHANGED_DURING_UPDATER"


class DivergentReplayWarning(AlganWarning):
    """Warns that a recorded call, re-run to render its frames, reached a Mob
    that did not exist when the call was recorded.

    Algan renders an animated function's frames by calling it again, with its
    recorded arguments interpolated for each frame. That only reproduces the
    animation if the call reaches the same Mobs every time it runs. A call that
    reads a mutable object -- one captured in its arguments, or a global --
    which the script changes after the call, reaches whatever that object
    holds by the time the frames are rendered: its frames then show a Mob the
    script made later, and the Mob it animated when it was recorded gets
    nothing. Stills and video frames show the same thing, since both render
    this way.
    """

    code = "ALGAN_DIVERGENT_REPLAY"


class Float32PrecisionWarning(AlganWarning):
    """Warns that float32 rounding moved geometry far enough to see on screen.

    Algan stores and renders positions in float32, about 7 significant
    digits, so a position is only as exact as its largest coordinate allows.
    Far from the origin that spacing can reach a pixel: typically a camera
    hundreds or thousands of units out, turned off its axes, with something
    placed just in front of it. The render estimates the rounding of every
    visible point after projecting it and warns once, naming the worst Mob,
    when it reaches half a pixel. Keeping the camera and what sits in front of
    it within a few hundred units of the origin avoids it.

    It also warns, once and naming both Mobs, when two overlapping flat shapes
    are closer in depth than a distant camera can tell apart: a few float32
    spacings of the camera's distance from the origin, 0.0076 units for an eye
    16,000 units out. Near the origin, shapes stacked at one depth are drawn
    in the order they were made; that far out rounding decides, and they can
    trade places from frame to frame. Separating them along the view, or
    bringing the camera closer, avoids it.
    """

    code = "ALGAN_FLOAT32_PRECISION"


__all__ = [
    "AlganError",
    "AlganConfigurationError",
    "UnsupportedFeatureError",
    "HierarchyError",
    "ModifiedProtectedAttributeError",
    "AudioTranscriptMismatchError",
    "InvalidColorError",
    "ContextReuseError",
    "AlganWarning",
    "UnsupportedFeatureWarning",
    "LegacySceneDiscoveryWarning",
    "ApproximationWarning",
    "NeverSpawnedMobWarning",
    "DespawnedMobWarning",
    "NeverVisibleMobWarning",
    "AnimatedBeforeSpawnWarning",
    "HierarchyChangedDuringUpdaterWarning",
    "DivergentReplayWarning",
    "Float32PrecisionWarning",
]
