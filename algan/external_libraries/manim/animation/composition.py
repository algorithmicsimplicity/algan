"""Inert stand-ins for Manim's animation containers."""

from __future__ import annotations

from .animation import Animation

__all__ = ["AnimationGroup", "LaggedStart", "Succession"]


class AnimationGroup(Animation):
    """See :mod:`~manim.animation`."""


class Succession(AnimationGroup):
    """See :mod:`~manim.animation`."""


class LaggedStart(AnimationGroup):
    """See :mod:`~manim.animation`."""
